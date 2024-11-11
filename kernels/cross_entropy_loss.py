# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by Juk Armstrong, November 2024:
# ==============================================
#
# Added support for Logit Scaling:
# - This allows scaling logits by a constant factor before computing the loss,
#   which can be useful for models requiring temperature scaling or adjusting
#   the sharpness of the softmax.
# - This also facilitates the implementation of Focal Loss:
#   https://arxiv.org/abs/1708.02002 (Appendix A and B).
#
# Added support for Label Smoothing:
# - The smoothed labels are: y_smooth = (1 - gamma) * y + gamma / (|V| - 1)
# - The full loss with label smoothing, where U ~ Uniform(1 / |V|):
#       Loss = CE_loss(y_smooth, p)
#            = (1 - gamma) * H(y, p) + gamma * H(U, p)
#            = (1 - gamma) * (-Σ y_i * log(p_i)) + gamma * (-Σ u_i * log(p_i))
# - For efficiency, we use the approximation:
#       Loss ≈ logsumexp - (1 - gamma) * z_target + gamma * log(|V| - 1)
#   This approximation avoids explicitly summing over all vocabulary tokens
#   for the uniform term and maintains similar optimization behaviour since
#   the value log(|V| - 1) is constant with respect to the parameters.
# - The backward pass computes the gradients as:
#       dL/dz_i = p_i - y_smooth_i
#   where y_smooth_i includes the label smoothing adjustments.

import triton
import triton.language as tl
import torch
from .utils import calculate_settings, MAX_FUSED_SIZE
from transformers.models.llama.modeling_llama import logger


@triton.heuristics({
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
    "DO_LABEL_SMOOTHING": lambda args: args["DO_LABEL_SMOOTHING"],
})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE_FACTOR: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    SMOOTHING_GAMMA: tl.constexpr,
):
    """
        Cross-Entropy Loss for a single sample is defined as:
            Loss = -y_target * log(p_target)
        Where:
            - y_target is the target label (1 for correct class, 0 otherwise).
            - p_target is the predicted probability for the target class.
        
        Given the logits z_i, the predicted probabilities p_i are computed using softmax:
            p_i = exp(z_i) / sum_j exp(z_j)
        
        Combining these, the loss can be rewritten:
            Loss = -log(p_target)
                 = -log(exp(z_target) / sum_j exp(z_j))
                 = - (z_target - logsumexp)
        
        Where:
            logsumexp = log(sum_j exp(z_j))
        
        To compute logsumexp in a numerically stable way, we use the trick:
            logsumexp = max_z + log(sum_j exp(z_j - max_z))
        
        When logit scaling is applied, each logit z_i is scaled by s:
            z_i = s * z_i
                
        With label smoothing, the targets y_i are modified:
            y_target = (1 - gamma)    for the true label
            y_i = gamma / (|V| - 1)   for all other labels i != t
        
        Instead of computing the full loss:
            Loss = (1-gamma)*(-log p_t) + gamma*(-Σ (1/|V|)*log p_i)
        
        We use:
            Loss ≈ logsumexp - (1-gamma)*z_target + gamma*log(|V| - 1)
        
        This avoids having to explicitly sum over the full vocabulary for the uniform
        distribution term,while maintaining equivalent optimization behaviour since
        the value of log(|V| - 1) is constant w.r.t the parameters.
    """
    row_idx = tl.program_id(0)
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx
    labels_ptr    += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE_FACTOR * logits

    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        logit_target = tl.load(logits_ptr + label_idx).to(tl.float32)
        if DO_LOGIT_SCALING:
            logit_target = LOGIT_SCALE_FACTOR * logit_target
        if DO_LABEL_SMOOTHING:
            loss = logsumexp - (1.0 - SMOOTHING_GAMMA) * logit_target + SMOOTHING_GAMMA * tl.log(float(VOCAB_SIZE - 1))
        else:
            loss = logsumexp - logit_target
    else:
        loss = 0.0

    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)


@triton.heuristics({
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
    "DO_LABEL_SMOOTHING": lambda args: args["DO_LABEL_SMOOTHING"],
})
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE_FACTOR: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    SMOOTHING_GAMMA: tl.constexpr,
):
    """
        For large VOCAB_SIZE > 65336, we divide into chunks to fit within MAX_FUSED_SIZE:
        
        First split the vocabulary into N_CHUNKS chunks, each of size BLOCK_SIZE.
        
        For each chunk, we compute:
            logsumexp_chunk = max_z_chunk + log(sum_i exp(z_i - max_z_chunk))
        
        After computing logsumexp for each chunk, we combine them:
            logsumexp_total = logsumexp(cat(logsumexp_chunks))
        
        This works because:
            exp(logsumexp_total) = sum_j exp(logsumexp_chunk_j) = sum_i exp(z_i)
        
        Thus, we obtain the overall logsumexp needed for the loss computation.
        
        The loss for each sample is then:
            Loss = logsumexp_total - z_target
        
        If label smoothing is applied:
            Loss = logsumexp_total - (1 - gamma) * z_target + gamma * log(|V| - 1)
    """
    row_idx   = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr    += row_idx

    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE_FACTOR * logits

    c = tl.max(logits, 0)
    logsumexp_chunk = c + tl.log(tl.sum(tl.exp(logits - c), 0))
    tl.store(logsumexp_ptr, logsumexp_chunk)

    if chunk_idx == 0:
        if label_idx != -100:
            logit_target = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING:
                logit_target = LOGIT_SCALE_FACTOR * logit_target
            if DO_LABEL_SMOOTHING:
                loss = - (1.0 - SMOOTHING_GAMMA) * logit_target + SMOOTHING_GAMMA * tl.log(float(VOCAB_SIZE - 1))
            else:
                loss = - logit_target
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)


@triton.heuristics({
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
    "DO_LABEL_SMOOTHING": lambda args: args["DO_LABEL_SMOOTHING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE_FACTOR: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    SMOOTHING_GAMMA: tl.constexpr,
):
    """
        The gradient of the cross-entropy loss with respect to the logits z_i is:
        
            dL/dz_i = p_i - y_i
        
        Where:
            - p_i = exp(z_i - logsumexp) is the predicted probability for class i.
            - y_i = target probability for class i (1 for the correct class, 0 otherwise).
        
        With label smoothing:
            - For the target class (i = target):
                y_i = 1 - gamma
            - For other classes:
                y_i = gamma / (|V| - 1)
        
        Thus, the gradient becomes:
            dL/dz_i = p_i - y_i
        
        If logit scaling is applied (z_i = s * z_i), then:
            dL/dz_i = s * (p_i - y_i)
    """
    row_idx   = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr  += row_idx * dloss_row_stride
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if DO_LOGIT_SCALING:
        logits = LOGIT_SCALE_FACTOR * logits

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    p = tl.exp(logits - logsumexp)

    if DO_LABEL_SMOOTHING:
        d_logits = tl.where(
            col_offsets == label_idx,
            p - (1.0 - SMOOTHING_GAMMA),
            p - SMOOTHING_GAMMA / float(VOCAB_SIZE - 1)
        )
    else:
        d_logits = tl.where(
            col_offsets == label_idx,
            p - 1.0,
            p
        )

    if DO_LOGIT_SCALING:
        d_logits = LOGIT_SCALE_FACTOR * d_logits

    # NOTE: The original values of `logits` overwritten (deliberate to save VRAM?).
    tl.store(logits_ptr + col_offsets, dloss * d_logits, mask=mask)


MAX_FUSED_SIZE = 65536 # 2**16

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale_factor = 1.0, smoothing_gamma = 0.0):
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

        if n_chunks == 1:
            # For small vocabs <= 65336 like Llama, Mistral
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype = torch.float32, device = "cuda")

            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE = vocab_size,
                BLOCK_SIZE = BLOCK_SIZE,
                DO_LOGIT_SCALING = (logit_scale_factor != 1.0),
                LOGIT_SCALE_FACTOR = logit_scale_factor,
                DO_LABEL_SMOOTHING = (smoothing_gamma != 0.0),
                SMOOTHING_GAMMA = smoothing_gamma,
                num_warps  = num_warps,
            )
        else:
            # For large vocabs > 65336 like Gemma 256K
            logsumexp = torch.empty((n_rows, n_chunks,), dtype = torch.float32, device = "cuda")

            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE = vocab_size,
                N_CHUNKS   = n_chunks,
                BLOCK_SIZE = MAX_FUSED_SIZE,
                DO_LOGIT_SCALING = (logit_scale_factor != 1.0),
                LOGIT_SCALE_FACTOR = logit_scale_factor,
                DO_LABEL_SMOOTHING = (smoothing_gamma != 0.0),
                SMOOTHING_GAMMA = smoothing_gamma,
                num_warps  = 32,
            )
            # logsumexp(chunked_logsumexp) - x
            # Do the -x separately
            logsumexp = torch.logsumexp(logsumexp, dim = 1) # Row sum
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0) # Don't forget to mask padding out!

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.logit_scale_factor = logit_scale_factor
        ctx.smoothing_gamma = smoothing_gamma
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        _cross_entropy_backward[(n_rows, n_blocks,)](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE = vocab_size,
            BLOCK_SIZE = BLOCK_SIZE,
            DO_LOGIT_SCALING = (ctx.logit_scale_factor != 1.0),
            LOGIT_SCALE_FACTOR = ctx.logit_scale_factor,
            DO_LABEL_SMOOTHING = (ctx.smoothing_gamma != 0.0),
            SMOOTHING_GAMMA = ctx.smoothing_gamma,
            num_warps  = 8,
        )
        return logits, None, None, None


def fast_cross_entropy_loss(
        logits,
        labels,
        logit_scale_factor = 1.0,
        smoothing_gamma = 0.0
):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        logit_scale_factor,
        smoothing_gamma,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
