from collections import deque

import torch
from torch import nn

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm as dist
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.pipe.engine import PipelineEngine, TRAIN_BATCH_TIMER, PIPE_SEND_OUTPUT_TIMER, PIPE_SEND_GRAD_TIMER, PIPE_RECV_INPUT_TIMER, PIPE_RECV_GRAD_TIMER
from deepspeed.runtime.pipe import schedule
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.pipe.module import PipelineModule
from deepspeed.runtime import utils as ds_utils
from deepspeed.runtime.pipe.module import LayerSpec

from utils import eta_str, log


def initialize(args=None,
               model=None,
               model_parameters=None,
               optimizer=None):
    assert model is not None, "deepspeed.initialize requires a model"

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(dist_backend=dist_backend)

    config = args.deepspeed_config
    assert config is not None, "DeepSpeed requires --deepspeed_config to specify configuration file"

    mpu = model.mpu()
    config_class = DeepSpeedConfig(config, mpu)
    engine = CustomPipelineEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        mpu=mpu,
        config=config,
        config_class=config_class
    )

    return engine, engine.optimizer


def compute_orthogonality_regularization(model):
    """
    Computes approximation of ||CᵗC - I||_F², where C = I + BA:
    - Full expansion is ||BA + AB + (BA)(AB)||_F²
    - For small-norm ||A||,||B|| we approximate using just ||BA + AB||_F²
    - This expands to ||BA||_F² + ||AB||_F² + 2⟨BA,AB⟩ = 2||AB||_F² + 2Tr(AAᵗBᵗB)
    
    Computationally, this approximation exploits that BA has rank ≤ k (k << n):
    - Approximate version is O(nk²), full expansion would be O(n³) due to (BA)(AB) term.
    - Avoids O(n³) operations of computing full ||CᵗC - I||²_F
    - Captures main effects in k-dim subspace spanned by A,B
    - Valid since BA≈0 outside this subspace due to low rank
    
    Numerically, this is a good approximation when ||A||,||B|| are small (< 1):
    - The two retained terms are both O(||A||⁴,||B||⁴)
    - The dropped (BA)(AB) term is O(||A||⁸,||B||⁸)
    - When ||A||,||B|| are large (>> 1), the dropped term starts to dominate though...
    
    The two retained terms have distinct interpretations:
    1. ||AB||_F²
       - Cross-covariance term penalising scaling
       - AB is the k x k cross-covariance matrix
       - Measures magnitude/energy of the k x k transformation AB
       - Large values indicate BA is scaling vectors up/down too much
    2. Tr(AAᵗBᵗB)
       - Covariance alignment term penalising shearing/skewing
       - AAᵗ and BᵗB are the k x k covariance matrices
       - Their product/trace measures non-orthogonal rotation effects
       - Large values mean BA introduces unwanted shearing/skewing
    """
    lora_scale = 1.0  # TODO: Pass in same way as orthogonality_lambda via ComputeMetrics

    total_norm = 0.0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            A = param                                   # k x n
            B_name = name.replace('lora_A', 'lora_B')
            B = next(p for n, p in model.named_parameters() if n == B_name)  # n x k
            
            AB = lora_scale * (A @ B)                   # k x k
            AB_norm_sq = torch.norm(AB, p='fro') ** 2
            AAt = lora_scale * (A @ A.T)                # k x k
            BtB = lora_scale * (B.T @ B)                # k x k
            trace_AAt_BtB = torch.trace(AAt @ BtB)
            E_norm_sq_approx = 2 * AB_norm_sq + 2 * trace_AAt_BtB
            
            total_norm += E_norm_sq_approx
            lora_count += 1

    if torch.isnan(total_norm):
        raise RuntimeError('NaN detected in norm calculation, probably some/all weights are NaN')
    
    # Return sum and count, as these will be accumulated over the pipeline stages 
    return total_norm, lora_count


# NOTE: This works, but trace((MN)^2) causes underflow for low-norm input matrices and
#       the output just ends up as `n` due to the `+ n` dominating...
def compute_orthogonality_regularization_exact_1(model):
    """
    Computes the exact value of ||CᵗC - I||_F² in an efficient manner, where C = I + BA.

    By expressing the norm in terms of smaller k × k matrices:
      - Compute M = A @ A.T (size k x k)
      - Compute N = B.T @ B (size k x k)
      - Compute MN = M @ N (size k x k)
      - Compute trace(MN) and trace((MN)^2)

    Then, the norm is computed as:
      ||CᵗC - I||_F² = trace((MN)^2) - 2 * trace(MN) + n

    This avoids operations on large n x n matrices and is efficient when k << n.
    """
    lora_scale = 1.0  # TODO: Pass in same way as orthogonality_lambda via ComputeMetrics
    total_norm = 0.0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            A = param  # k x n
            B_name = name.replace('lora_A', 'lora_B')
            B = next(p for n, p in model.named_parameters() if n == B_name)  # n x k

            # Apply scaling if necessary
            A_scaled = lora_scale * A
            B_scaled = lora_scale * B

            # Compute M and N (k x k matrices)
            M = A_scaled @ A_scaled.T  # k x k
            N = B_scaled.T @ B_scaled  # k x k

            # Compute MN and its square
            MN = M @ N  # k x k
            MN_squared = MN @ MN  # k x k

            # Compute the traces
            trace_MN = torch.trace(MN)
            trace_MN_squared = torch.trace(MN_squared)

            # Compute n, the size of the identity matrix I
            n = B.shape[0]

            # Compute the exact norm
            E_norm_sq_exact = trace_MN_squared - 2 * trace_MN + n

            total_norm += E_norm_sq_exact
            lora_count += 1

    if torch.isnan(total_norm):
        raise RuntimeError('NaN detected in norm calculation, probably some/all weights are NaN')

    # Return sum and count, as these will be accumulated over the pipeline stages
    return total_norm, lora_count


# NOTE: This works but has to materialise the n x n matrices...
def compute_orthogonality_regularization_exact_2(model):
    """
    Computes the exact value of ||CᵗC - I||_F² using an alternative computation
    that avoids numerical underflow by directly computing small terms.

    Method:
        - Compute BA = B @ A (size n x n)
        - Compute BA_symm = BA + BA.T + BA.T @ BA
        - Compute ||BA_symm||_F²

    Note: This method involves computations with large n x n matrices, which may
    increase computational cost and memory usage when n is large.
    """
    lora_scale = 1.0  # TODO: Pass in same way as orthogonality_lambda via ComputeMetrics
    total_norm = 0.0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            A = param  # k x n
            B_name = name.replace('lora_A', 'lora_B')
            B = next(p for n, p in model.named_parameters() if n == B_name)  # n x k

            # Apply scaling if necessary
            A_scaled = lora_scale * A
            B_scaled = lora_scale * B

            # Compute BA (n x n matrix)
            BA = B_scaled @ A_scaled  # n x n

            # Compute BA_symm = BA + BA.T + BA.T @ BA
            BA_symm = BA + BA.T + BA.T @ BA  # n x n

            # Compute the Frobenius norm squared
            E_norm_sq_exact = torch.norm(BA_symm, p='fro') ** 2

            total_norm += E_norm_sq_exact.item()
            lora_count += 1

    if torch.isnan(torch.tensor(total_norm)):
        raise RuntimeError('NaN detected in norm calculation, probably some/all weights are NaN')

    # Return sum and count, as these will be accumulated over pipeline stages
    return total_norm, lora_count

# This almost works, but between 0.1 and 10 we get slight instability compared to naive method...
def compute_orthogonality_regularization_exact_3(model):
    """
    Computes ||CᵗC - I||_F² using a dynamic approach:
    - Computes trace(MN) and determines whether numerical underflow is likely.
    - If underflow is detected (trace(MN) is below threshold), switches to the approximate method.
    - Otherwise, uses the efficient exact method.
    """
    lora_scale = 1.0  # Adjust as needed
    total_norm = 0.0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            A = param  # Adjust dimensions if needed
            B_name = name.replace('lora_A', 'lora_B')
            B = next(p for n, p in model.named_parameters() if n == B_name)

            # Apply scaling as necessary
            A_scaled = lora_scale * A
            B_scaled = lora_scale * B

            # Compute M and N (k x k matrices)
            M = A_scaled @ A_scaled.T  # k x k
            N = B_scaled.T @ B_scaled  # k x k

            # Compute MN and its square
            MN = M @ N
            MN_squared = MN @ MN

            # Compute traces
            trace_MN = torch.trace(MN)
            trace_MN_squared = torch.trace(MN_squared)

            # Check for potential underflow
            if trace_MN.abs().item() < 1:
                # Underflow likely, switch to approximate method
                AB = A_scaled @ B_scaled  # Adjust dimensions if needed
                AB_norm_sq = torch.norm(AB, p='fro') ** 2
                AAt = A_scaled @ A_scaled.T
                BtB = B_scaled.T @ B_scaled
                trace_AAt_BtB = torch.trace(AAt @ BtB)
                E_norm_sq = 2 * AB_norm_sq + 2 * trace_AAt_BtB
            else:
                # Compute exact norm
                n = B_scaled.shape[0]
                E_norm_sq = trace_MN_squared - 2 * trace_MN + n

            total_norm += E_norm_sq.item()
            lora_count += 1

    if torch.isnan(torch.tensor(total_norm)):
        raise RuntimeError('NaN detected in norm calculation, possibly due to numerical issues')

    return total_norm, lora_count


def compute_lp_regularization(model, p = 2):
    """Computes Lp-Regularisation (aka "Power Ridge Regression" or "Bridge Regression")
    
    Args:
        model: The model containing LoRA weights
        p: Power for the norm, must be >= 1 (default=2)
           p=2 gives Ridge Regression, p=1 gives Lasso
    
    See: https://www.stat.cmu.edu/technometrics/90-00/vol-35-02/v3502109.pdf
    """
    assert p >= 1, "p<1 is non-convex and not suitable for gradient-based optimization methods"
    lora_scale = 1.0  # TODO: Pass in same way as orthogonality_lambda via ComputeMetrics
    
    total_norm = 0.0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora_A' in name:
            A = param                                        # k x m
            B_name = name.replace('lora_A', 'lora_B')
            B = next(p for n, p in model.named_parameters() if n == B_name)  # n x k

            if p == 2:
                # Special case for p=2: use efficient trace method
                AAt = lora_scale * (A @ A.T)                 # k x k
                BAAt = B @ AAt                               # n x k
                norm = lora_scale * torch.trace(BAAt @ B.T)
            else:
                # For other p values, need to materialise B @ A
                BA = lora_scale * (B @ A)                    # n x m
                norm = torch.sum(torch.abs(BA) ** p)

            # Divide by p to cancel out the derivative pulling the power down.
            total_norm += (1.0 / p) * norm
            lora_count += 1

    if torch.isnan(total_norm):
        raise RuntimeError('NaN detected in norm calculation, probably some/all weights are NaN')

    # Return sum and count, as these will be accumulated over the pipeline stages 
    return total_norm, lora_count


class CustomPipelineEngine(PipelineEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_steps = None
        self.etas = deque()


    def train_batch(self):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.train()
        self._compute_loss = True

        # Do the work
        self.timers(TRAIN_BATCH_TIMER).start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        agg_losses = self._aggregate_total_losses()
        # Actual training loss is always the first item.
        self.agg_train_loss = agg_losses[0].mean()

        self.timers(TRAIN_BATCH_TIMER).stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                eta = iter_time * (self.total_steps - self.global_steps)
                self.etas.append(eta)
                while len(self.etas) > 10:
                    self.etas.popleft()
                rolling_eta = sum(self.etas) / len(self.etas)
                tput = self.train_batch_size() / iter_time
                log(f'step: {self.global_steps:>5} / {self.total_steps:>5} '
                    f'loss: {self.agg_train_loss:0.4f} '
                    f'iter time (s): {iter_time:0.3f} '
                    f'samples/sec: {tput:0.3f} '
                    f'eta: {eta_str(rolling_eta)} ')
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log([
                PIPE_SEND_OUTPUT_TIMER,
                PIPE_SEND_GRAD_TIMER,
                PIPE_RECV_INPUT_TIMER,
                PIPE_RECV_GRAD_TIMER,
            ])

        return agg_losses


    def eval_batch(self, data_iter):
        # sequence length may change between macro batches (but not between gradient accumulation steps)
        self.reset_activation_shape()

        self.module.eval()
        self._compute_loss = True

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        with torch.no_grad():
            self._exec_schedule(sched)

        # list of losses
        agg_eval_losses = self._aggregate_total_losses()

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/eval_loss', agg_eval_losses[0].mean().item(), self.global_samples)]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)

        return agg_eval_losses


    def _aggregate_total_losses(self):
        all_agg_outputs = []
        # gather each output for all the gradient accumulation steps
        grouped_outputs = [list(x) for x in zip(*self.fwd_outputs)]
        # if any are scalar, make them dim 1 so we can concat across DP ranks
        for outputs in grouped_outputs:
            for i, output in enumerate(outputs):
                if output.dim() == 0:
                    outputs[i] = torch.unsqueeze(output, 0)

        if self.is_last_stage():
            agg_sizes = []
            # loop to gather all the outputs across DP ranks
            for outputs in grouped_outputs:
                # concat all the grad_accum_steps
                concat_outputs = torch.cat(outputs)
                if self.is_data_parallel:
                    # might be different sizes across DP ranks, so, gather all the sizes
                    sizes = [None] * self.grid.get_data_parallel_world_size()
                    torch.distributed.all_gather_object(sizes, concat_outputs.size(), group=self.grid.get_data_parallel_group())
                    # once we know all the sizes we can gather the results across DP ranks
                    gather_result = [torch.zeros(size).to(self.device) for size in sizes]
                    dist.all_gather(gather_result, concat_outputs, group=self.grid.get_data_parallel_group())
                    # and finally, concat
                    agg_output = torch.cat(gather_result)
                else:
                    agg_output = concat_outputs
                agg_sizes.append(agg_output.size())
                all_agg_outputs.append(agg_output)

            # send the sizes, then broadcast to the PP ranks
            if self.is_pipe_parallel:
                torch.distributed.broadcast_object_list([agg_sizes], src=self.global_rank, group=self.grid.get_pipe_parallel_group())
                for agg_output in all_agg_outputs:
                    dist.broadcast(tensor=agg_output, src=self.global_rank, group=self.grid.get_pipe_parallel_group())
        else:
            # get the outputs from the last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            result = [None]
            torch.distributed.broadcast_object_list(result, src=src_rank, group=self.grid.get_pipe_parallel_group())
            agg_sizes = result[0]
            for agg_size in agg_sizes:
                agg_output = torch.zeros(agg_size).to(self.device)
                dist.broadcast(tensor=agg_output, src=src_rank, group=self.grid.get_pipe_parallel_group())
                all_agg_outputs.append(agg_output)

        return all_agg_outputs


    # We override this to handle the model returning a list of "losses", but only doing backprop on the first.
    ### JUK: Now also used to calculate the regularisation terms over multiple nodes and stages too (I THINK!?).
    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)
   
        if isinstance(self.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][buffer_id].clone()
    
        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())
    
            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            part_input = None
            # Check if orthogonality regularization is included in inputs
            if len(inputs) >= 2 and all(isinstance(t, torch.Tensor) and t.dim() == 0 for t in inputs[-2:]):
                ortho_reg_prev_norm = inputs[-2]
                ortho_reg_prev_count = inputs[-1]
                inputs = inputs[:-2]
            else:
                ortho_reg_prev_norm = torch.tensor(0.0, device=self.device)
                ortho_reg_prev_count = torch.tensor(0.0, device=self.device)
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][buffer_id] = inputs
        else:
            ortho_reg_prev_norm = torch.tensor(0.0, device=self.device)
            ortho_reg_prev_count = torch.tensor(0.0, device=self.device)
    
        # inputs has no gradient because it is from a cloned tensor
        outputs = super(PipelineEngine, self).forward(inputs)
    
        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()
    
        # Compute orthogonality regularization
        orthogonality_lambda = 0
        if hasattr(self.module, '_layer_specs'):
            orthogonality_lambda = self.module._layer_specs[-1].module_kwargs.get('orthogonality_lambda', 0)
        
        # Have each rank compute the regularization term independently on its own collection of tensors
        if orthogonality_lambda > 0:
            ortho_reg_norm, ortho_reg_count = compute_orthogonality_regularization(self.module)
        else:
            ortho_reg_norm = torch.tensor(0.0, device=self.device)
            ortho_reg_count = torch.tensor(0.0, device=self.device)
        
        # Accumulate orthogonality regularization
        total_ortho_reg_norm = ortho_reg_prev_norm + ortho_reg_norm
        total_ortho_reg_count = ortho_reg_prev_count + ortho_reg_count
    
        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None
    
        # Include the accumulated orthogonality regularization in outputs
        if isinstance(outputs, tuple):
            outputs = outputs + (total_ortho_reg_norm, total_ortho_reg_count)
        else:
            outputs = (outputs, total_ortho_reg_norm, total_ortho_reg_count)
    
        self.pipe_buffers['outputs'][buffer_id] = outputs
    
        # Optionally compute loss on the last device
        if self.is_last_stage():
            # Extract total_ortho_reg_norm and total_ortho_reg_count
            total_ortho_reg_norm = outputs[-2]
            total_ortho_reg_count = outputs[-1]
            outputs = outputs[:-2]
            
            # Compute mean_ortho_reg, adding a small epsilon to avoid division by zero
            mean_ortho_reg = total_ortho_reg_norm / (total_ortho_reg_count + 1e-8)
    
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers['labels'][buffer_id]
                losses = self.module.loss_fn(outputs, labels)
            else:
                # Some models just return loss from forward()
                losses = outputs
    
            # Add orthogonality regularization
            if orthogonality_lambda > 0:
                if isinstance(losses, torch.Tensor):
                    losses = losses + orthogonality_lambda * mean_ortho_reg
                else:
                    losses = (losses[0] + orthogonality_lambda * mean_ortho_reg, *losses[1:])
         
            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(losses, torch.Tensor):
                self.loss = losses
                self.fwd_outputs.append([self.loss.detach()])
            else:
                self.loss = losses[0]
                self.fwd_outputs.append([l.detach() for l in losses])
    
    # make our forward pass method apply
    PipelineEngine._INSTRUCTION_MAP[schedule.ForwardPass] = _exec_forward_pass
   

class CustomPipelineModule(PipelineModule):
    def __init__(self, layers, **kwargs):
        super().__init__(layers, **kwargs)

    def _partition_layers(self, method='uniform'):
        num_stages = self._topo.get_dim('pipe')
        stage_id = self._topo.get_coord(self.global_rank).pipe

        if self.global_rank == 0:
            print(f'Partitioning pipeline stages with method {method}')

        method = method.lower()

        estimated_sizes = None
        # Each stage gets a simple uniform number of layers.
        if method == 'uniform':
            num_layers = len(self._layer_specs)
            self.parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
        elif method == 'parameters':
            param_counts = self._count_layer_params()
            self.parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
        elif method.startswith('type:'):
            layertype = method.split(':')[1]
            binary_weights = [0] * len(self._layer_specs)
            for idx in self._find_layer_type(layertype):
                binary_weights[idx] = 1
            self.parts = ds_utils.partition_balanced(weights=binary_weights, num_parts=num_stages)
        elif method == 'profile':
            raise NotImplementedError(f'Partitioning method {method} not implemented.')
        elif method == 'estimated_size':
            estimated_sizes = [getattr(l, 'estimated_size', 0) for l in self._layer_specs]
            self.parts = ds_utils.partition_balanced(weights=estimated_sizes, num_parts=num_stages)
        else:
            raise NotImplementedError(f'Partitioning method {method} not implemented.')

        # Print some information on the partitioning.
        if self.global_rank == 0:
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                print(f'stage={stage} layers={stop - start}')
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, LayerSpec):
                        name = layer.typename.__name__
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    logstr = f'    {idx+start:2d}: {name}'
                    if estimated_sizes:
                        es = estimated_sizes[idx+start]
                        logstr += f', estimated size: {es}'
                    print(logstr)
            if self.loss_fn:
                try:
                    print(f'  loss: {self.loss_fn.__name__}')
                except AttributeError:
                    print(f'  loss: {self.loss_fn.__class__.__name__}')
        deepspeed.comm.barrier()

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])
