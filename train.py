import argparse
import os
from datetime import datetime, timezone
import shutil
import glob
import time
import itertools
from contextlib import contextmanager
import json
import gc

import torch
from torch.utils.tensorboard import SummaryWriter
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import deepspeed
from deepspeed.runtime.pipe.module import PipelineModule, LayerSpec
import toml
import bitsandbytes
from safetensors.torch import save_file

from dataset_utils import load_datasets
import dataloader
from utils import *
import engine
import llama_pipe
import mixtral_pipe

DTYPE_MAP = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--debug_dataset', type=int, help='print out this many training examples and then quit')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def write_metrics(tb_writer, prefix, metrics, step):
    losses = metrics[1].view(-1)
    sorted_losses, sorted_losses_idx = torch.sort(losses)
    quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999], dtype=torch.float32).to(losses.device)
    quantiles_idx = [int(len(losses)*quantile) for quantile in quantiles]
    loss_quantiles = [sorted_losses[i] for i in quantiles_idx]
    for quantile, value in zip(quantiles, loss_quantiles):
        tb_writer.add_scalar(f'{prefix}/loss_quantile_{quantile:.3f}', value, step)
    tb_writer.add_scalar(f'{prefix}/loss', losses.mean().item(), step)
    tb_writer.add_scalar(f'{prefix}/optimized_loss', metrics[0].mean().item(), step)
    tb_writer.add_histogram(f'{prefix}/log_loss_hist', torch.log(1e-10 + losses), step)

    entropy = metrics[2].view(-1)
    assert entropy.size() == losses.size()
    tb_writer.add_scalar(f'{prefix}/entropy', entropy.mean().item(), step)
    sorted_entropy = entropy[sorted_losses_idx]
    entropy_quantiles = []
    for i, j in itertools.zip_longest(quantiles_idx, quantiles_idx[1:]):
        entropy_quantiles.append(sorted_entropy[i:j].mean())
    for quantile, value in zip(quantiles, entropy_quantiles):
        tb_writer.add_scalar(f'{prefix}/entropy_quantile_{quantile:.3f}', value, step)

    tb_writer.add_scalar(f'{prefix}/top1_accuracy', metrics[3].mean().item(), step)
    tb_writer.add_scalar(f'{prefix}/top5_accuracy', metrics[4].mean().item(), step)
    tb_writer.add_scalar(f'{prefix}/top20_accuracy', metrics[5].mean().item(), step)

    if len(metrics) >= 7:
        tb_writer.add_scalar(f'{prefix}/load_balancing_loss', metrics[6].mean().item(), step)
    if len(metrics) >= 8:
        tb_writer.add_scalar(f'{prefix}/alternate_load_balancing_loss', metrics[7].mean().item(), step)


def evaluate_single(model_engine, name, eval_dataloader, tb_writer, step, eval_gradient_accumulation_steps):
    orig_micro_batches = model_engine.micro_batches
    model_engine.micro_batches = eval_gradient_accumulation_steps
    iterator = iter(eval_dataloader)
    all_metrics = None
    while True:
        metrics = model_engine.eval_batch(iterator)
        eval_dataloader.sync_epoch()
        if all_metrics is None:
            all_metrics = [[] for _ in range(len(metrics))]
        if eval_dataloader.epoch == 2:
            break
        for i, metric in enumerate(metrics):
            all_metrics[i].append(metric)

    eval_dataloader.reset()
    model_engine.micro_batches = orig_micro_batches
    eval_metrics = [torch.cat(metric_list) for metric_list in all_metrics]
    if is_main_process():
        write_metrics(tb_writer, f'eval/{name}', eval_metrics, step)


def evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    if is_main_process():
        print('Running eval')
    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        evaluate_single(model_engine, name, eval_dataloader, tb_writer, step, eval_gradient_accumulation_steps)
    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)


# TODO: this is pretty hacky. Is there a way to get the state_dict from the lora model directly,
# but still know which layers the given pipeline parallel stage actually trained?
def save_lora(model_engine, pipeline_model, lora_config, save_dir, args):
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    if dp_id == 0:
        partial_state_dict = {}
        for name, p in pipeline_model.named_parameters():
            if p.requires_grad:
                if not hasattr(p, 'original_name'):
                    print(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                    continue
                partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = p
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
    deepspeed.comm.barrier()
    if dp_id == 0 and stage_id == 0:
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu'))
        torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
        lora_config.save_pretrained(save_dir)
        shutil.copy(args.config, save_dir)
        shutil.copy(args.deepspeed_config, save_dir)
        shutil.rmtree(tmp_dir)


def save_full_model(model_engine, pipeline_model, save_dir, args, config, max_shard_size='5GB'):
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    if dp_id == 0:
        partial_state_dict = {p.original_name: p for p in pipeline_model.parameters()}
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
    deepspeed.comm.barrier()
    if dp_id == 0 and stage_id == 0:
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu'))
        shards, index = transformers.modeling_utils.shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name='model.safetensors')
        for shard_file, shard in shards.items():
            save_file(shard, os.path.join(save_dir, shard_file), metadata={"format": "pt"})
        if index is not None:
            save_index_file = 'model.safetensors.index.json'
            save_index_file = os.path.join(save_dir, save_index_file)
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
        shutil.copy(args.config, save_dir)
        shutil.copy(args.deepspeed_config, save_dir)
        additional_files_to_copy = [
            'added_tokens.json',
            'config.json',
            'generation_config.json',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'tokenizer.model',
        ]
        for path in glob.glob(os.path.join(config['model'], '*')):
            if os.path.basename(path) in additional_files_to_copy:
                shutil.copy(path, save_dir)
        shutil.rmtree(tmp_dir)


def save_model(model_engine, pipeline_model, lora_config, save_dir, args, config):
    if lora_config is None:
        save_full_model(model_engine, pipeline_model, save_dir, args, config)
    else:
        save_lora(model_engine, pipeline_model, lora_config, save_dir, args)


def apply_max_norm_regularization(model, config):
    # modifed from https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
    A_keys = []
    B_keys = []
    norms = []
    keys_scaled = 0
    lora_scale = config['lora_alpha'] / config['lora_rank']

    state_dict = model.state_dict()
    for key in state_dict.keys():
        if 'lora_A' in key:
            A_keys.append(key)
            B_keys.append(key.replace('lora_A', 'lora_B'))

    for i in range(len(A_keys)):
        A = state_dict[A_keys[i]]
        B = state_dict[B_keys[i]]
        W = B @ A
        W *= lora_scale

        if 'scale_weight_norms' in config:
            max_norm = config['scale_weight_norms']
            norm = W.norm().clamp(min=max_norm / 2)
            desired = torch.clamp(norm, max=max_norm)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[A_keys[i]] *= sqrt_ratio
                state_dict[B_keys[i]] *= sqrt_ratio
        else:
            ratio = 1.0
        scalednorm = W.norm() * ratio
        norms.append(scalednorm.item())

    if len(norms) > 0:
        norms = torch.tensor(norms, dtype=torch.float32)
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
    else:
        avg_norm = 0
        max_norm = 0
    return keys_scaled, avg_norm, max_norm, norms


def parse_layers_to_transform(spec):
    parts = spec.split(',')
    result = []
    for part in parts:
        start, stop = part.split(':')
        result.extend(range(int(start), int(stop)+1))
    return result


@contextmanager
def one_at_a_time():
    for i in range(int(os.environ['LOCAL_SIZE'])):
        if i == int(os.environ['LOCAL_RANK']):
            yield
        deepspeed.comm.barrier()


def load_pipeline_model_with_lora(config):
    full_fine_tune = 'full_fine_tune' in config and config['full_fine_tune']
    with open(os.path.join(config['model'], 'config.json')) as f:
        model_config = json.load(f)
        model_type = model_config['model_type'] if 'model_type' in model_config else 'llama'

    if 'load_in_4bit' in config and config['load_in_4bit']:
        assert not full_fine_tune
        no_quant_modules = ['lm_head']
        if model_type == 'mixtral':
            # the expert routing weights are tiny and probably important, don't quantize
            no_quant_modules.append('gate')
        quantization_config_params = {
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': DTYPE_MAP[config['bnb_compute_dtype']],
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_use_double_quant': config['use_double_quant'],
            'llm_int8_skip_modules': no_quant_modules
        }
        quantization_config = transformers.BitsAndBytesConfig(**quantization_config_params)
    else:
        quantization_config = None

    focal_loss_gamma = 0 if 'focal_loss_gamma' not in config else config['focal_loss_gamma']
    if focal_loss_gamma > 0 and is_main_process():
        print(f'Using focal loss with gamma={focal_loss_gamma}')
    if model_type == 'llama' or model_type == 'mistral':
        model = llama_pipe.LlamaForCausalLMPipe(config['model'], quantization_config=quantization_config, focal_loss_gamma=focal_loss_gamma)
    elif model_type == 'mixtral':
        model = mixtral_pipe.MixtralForCausalLMPipe(
            config['model'],
            quantization_config=quantization_config,
            load_balancing_loss_coef=config['load_balancing_loss_coef'] if 'load_balancing_loss_coef' in config else None,
            focal_loss_gamma=focal_loss_gamma
        )
    elif model_type == 'qwen2':
        model = llama_pipe.Qwen2ForCausalLMPipe(config['model'], quantization_config=quantization_config, focal_loss_gamma=focal_loss_gamma)
    else:
        raise NotImplementedError()

    # CAREFUL! The "primary" layers of the model have to have 'decoderlayer' in them for
    # activation checkpointing to automatically work correctly.
    layers = model.to_layer_specs()
    checkpointable_layers = set()
    for layer in layers:
        if isinstance(layer, LayerSpec) and 'decoderlayer' in layer.typename.__name__.lower():
            checkpointable_layers.add(layer.typename.__name__)
    checkpointable_layers = list(checkpointable_layers)

    partition_method = 'uniform' if full_fine_tune else 'type:decoderlayer|embedding|lmhead'
    if config['activation_checkpointing']:
        pipeline_model = PipelineModule(
            layers=layers,
            num_stages=config['pipeline_stages'],
            activation_checkpoint_interval=1,
            checkpointable_layers=checkpointable_layers,
            activation_checkpoint_func=deepspeed.checkpointing.checkpoint,
            partition_method=partition_method
        )
    else:
        pipeline_model = PipelineModule(
            layers=layers,
            num_stages=config['pipeline_stages'],
            partition_method=partition_method
        )

    if quantization_config is not None:
        prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
        # convert them back to fp16/bf16 for flash-attn compatibility.
        for name, module in model.named_modules():
            dtype = DTYPE_MAP[config['bnb_compute_dtype']]
            if "norm" in name or "gate" in name:
                module.to(dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module.to(dtype)

    target_modules = config['target_modules'] if 'target_modules' in config else 'all-linear'
    if full_fine_tune:
        lora_model = None
        lora_config = None
        for name, p in model.named_parameters():
            p.original_name = name
        if isinstance(target_modules, list):
            for name, p in pipeline_model.named_parameters():
                if not any(target in name for target in config['target_modules']):
                    p.requires_grad = False
                    print(f'not training {name} because it is not present in target_modules')
    else:
        layers_to_transform = parse_layers_to_transform(config['layers_to_transform']) if 'layers_to_transform' in config else None
        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            target_modules=target_modules,
            modules_to_save=config['modules_to_save'] if 'modules_to_save' in config else [],
            lora_dropout=config['lora_dropout'] if 'lora_dropout' in config else 0,
            layers_to_transform=layers_to_transform,
            bias='none',
            task_type='CAUSAL_LM',
        )

        # If we set the default dtype to bfloat16 at the very beginning, the loss blows up.
        # If we set it only here for the lora weights, everything is fine. ¯\_(ツ)_/¯
        torch.set_default_dtype(DTYPE_MAP[config['lora_weight_dtype']])
        lora_model = get_peft_model(model, lora_config)
        torch.set_default_dtype(torch.float32)
        lora_model.model.config.use_cache = False
        for name, p in lora_model.named_parameters():
            p.original_name = name

    return pipeline_model, lora_model, lora_config


last_checkpoint_time = None
def need_to_checkpoint():
    global last_checkpoint_time
    checkpoint = False
    # rank 0 tracks if we need to checkpoint, broadcasts to everyone else
    if is_main_process():
        current_time = time.time()
        if last_checkpoint_time is None:
            last_checkpoint_time = current_time
        elif (current_time - last_checkpoint_time) / 60 > config['checkpoint_every_n_minutes']:
            checkpoint = True
            last_checkpoint_time = current_time
    result = [checkpoint]
    torch.distributed.broadcast_object_list(result, src=0)
    return result[0]


if __name__ == '__main__':
    # TODO: if resuming from checkpoint, probably should read all config files from checkpoint dir
    # rather than assume they are unchanged on the command line
    with open(args.config) as f:
        config = toml.load(f)

    deepspeed.init_distributed()

    # if this is a new run, create a new dir for it
    if not config['resume_from_checkpoint'] and is_main_process():
        run_dir = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(args.config, run_dir)
        shutil.copy(args.deepspeed_config, run_dir)
    # wait for all processes then get the most recent dir (may have just been created)
    deepspeed.comm.barrier()
    run_dir = get_most_recent_run_dir(config['output_dir'])

    tokenizer = transformers.AutoTokenizer.from_pretrained(config['model'], local_files_only=True, use_fast=False, add_bos_token=True, add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token

    train_data, eval_data_map = load_datasets(config, tokenizer)

    if args.debug_dataset:
        if is_main_process():
            for i, item in enumerate(iter(train_data)):
                print('decoded input_ids:')
                print(tokenizer.decode(item['input_ids']))
                print('attention_mask:')
                print(item['attention_mask'])
                print('labels:')
                print(item['labels'])
                print('-'*80)
                if i >= args.debug_dataset-1:
                    break
        quit()

    # for testing
    # train_data = train_data.select(list(range(100)))
    # eval_data = eval_data.select(list(range(50)))

    # Ugly hack so we can move quantized models from GPU to CPU, and back to GPU again without triggering quantization a second time.
    bnb_cuda_old = bitsandbytes.nn.modules.Params4bit.cuda
    def bnb_cuda_hijack(self, device):
        if getattr(self, 'already_quantized', False):
            self.data = self.data.to(device)
            self.quant_state.to(device)
            return self
        self.already_quantized = True
        return bnb_cuda_old(self, device)
    bitsandbytes.nn.modules.Params4bit.cuda = bnb_cuda_hijack

    pipeline_model, lora_model, lora_config = load_pipeline_model_with_lora(config)

    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    model_engine, optimizer = engine.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
    )

    # TODO: the main DeepSpeedEngine forces all parameters to the GPU, and also does things like
    # broadcast all parameters from data parallel rank 0 to all other ranks. Thus, MLP offloading
    # must come after engine.initialize(). If we want to avoid loading everything onto GPUs only
    # to offload the MLPs, we have to rewrite a lot of code to work around things.
    if 'offload_mlp_to_cpu' in config and config['offload_mlp_to_cpu']:
        assert config['activation_checkpointing']  # MLP offloading only works with activation checkpointing
        #pipeline_model.offload_mlp_to_cpu()
        for module in pipeline_model.modules():
            if hasattr(module, 'offload_mlp_to_cpu'):
                module.offload_mlp_to_cpu()
        torch.cuda.empty_cache()

    train_dataloader = dataloader.PipelineDataLoader(
        train_data,
        tokenizer,
        model_engine.train_micro_batch_size_per_gpu(),
        model_engine.gradient_accumulation_steps(),
        model_engine.grid.get_data_parallel_world_size(),
        model_engine.grid.get_data_parallel_rank(),
        group_by_length=False if 'group_by_length' not in config else config['group_by_length'],
        batch_size_tokens=None if 'batch_size_tokens' not in config else config['batch_size_tokens'],
    )
    model_engine.set_dataloader(train_dataloader)

    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()

    if 'lr_scheduler' not in config or config['lr_scheduler'] == 'constant' or config['lr_scheduler'] == 'none':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif config['lr_scheduler'] == 'cosine':
        total_steps = steps_per_epoch * config['epochs']
        total_steps -= config['warmup_steps'] if 'warmup_steps' in config else 0
        # Normally, you would pass the lr_scheduler to deepspeed.initialize(). But we need the
        # global batch_size in order to make the lr_scheduler.
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    else:
        raise NotImplementedError()

    if 'warmup_steps' in config:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])

    model_engine.lr_scheduler = lr_scheduler

    step = 1
    if config['resume_from_checkpoint']:
        load_path, client_state = model_engine.load_checkpoint(run_dir, load_module_strict=False, load_lr_scheduler_states=True)
        deepspeed.comm.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    # this is a separate option, because if it's too high we might drop a significant fraction of the eval dataset
    eval_gradient_accumulation_steps = config['eval_gradient_accumulation_steps'] if 'eval_gradient_accumulation_steps' in config else 1
    # Eval dataset doesn't need to repeat; we just use this to track "epoch" so we know when we're done iterating over it.
    eval_dataloaders = {
        name: dataloader.PipelineDataLoader(
            eval_data,
            tokenizer,
            model_engine.train_micro_batch_size_per_gpu(),
            eval_gradient_accumulation_steps,
            model_engine.grid.get_data_parallel_world_size(),
            model_engine.grid.get_data_parallel_rank(),
            shuffle=False,
            group_by_length=False if 'group_by_length' not in config else config['group_by_length'],
            # If we drop_last, we may lose up to batch_size*num_replicas data points. If we don't drop_last, we may have up
            # to an extra num_replicas data points as padding (and the last batch may be smaller). For a small dataset where
            # the batch_size doesn't affect any dynamics (since it's eval), the latter seems better.
            # TODO: drop_last=False still broken with pipelining, need to fix
            drop_last=True,
            batch_size_tokens=None if 'batch_size_tokens' not in config else config['batch_size_tokens'],
        )
        for name, eval_data in eval_data_map.items()
    }

    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None

    epoch = train_dataloader.epoch
    if config['eval_before_first_step'] and not config['resume_from_checkpoint']:
        evaluate(model_engine, eval_dataloaders, tb_writer, 0, eval_gradient_accumulation_steps)

    while True:
        gc.collect()
        torch.cuda.empty_cache()
        metrics = model_engine.train_batch()
        train_dataloader.sync_epoch()
        if lora_config is not None:
            keys_scaled, avg_norm, max_norm, norms = apply_max_norm_regularization(pipeline_model, config)

        if train_dataloader.epoch != epoch:
            model_engine.save_checkpoint(
                run_dir,
                client_state={
                    'step': step,
                    'custom_loader': train_dataloader.state_dict(),
                },
                save_latest=True,
                exclude_frozen_parameters=True
            )
            save_model(model_engine, pipeline_model, lora_config, f'{run_dir}/epoch{epoch}', args, config)
            epoch = train_dataloader.epoch
            if epoch > config['epochs']:
                break
            if is_main_process():
                print(f'Started new epoch: {epoch}')

        if is_main_process() and step % config['logging_steps'] == 0:
            write_metrics(tb_writer, 'train', metrics, step)
            tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            # TODO: gather the weight norms across all stages in the pipelined model, not just the first.
            if lora_config is not None and len(norms) > 0:
                tb_writer.add_scalar('train/weights_scaled', keys_scaled, step)
                tb_writer.add_scalar('train/avg_weight_norm', avg_norm, step)
                tb_writer.add_scalar('train/max_weight_norm', max_norm, step)
                tb_writer.add_histogram('train/weight_norm_hist', norms, step)
            tb_writer.add_scalar('train/epoch', step/steps_per_epoch, step)

        if step % config['save_steps'] == 0:
            save_model(model_engine, pipeline_model, lora_config, f'{run_dir}/step{step}', args, config)

        if step % config['eval_steps'] == 0:
            evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)

        if need_to_checkpoint():
            model_engine.save_checkpoint(
                run_dir,
                client_state={
                    'step': step,
                    'custom_loader': train_dataloader.state_dict(),
                },
                save_latest=True,
                exclude_frozen_parameters=True
            )

        step += 1