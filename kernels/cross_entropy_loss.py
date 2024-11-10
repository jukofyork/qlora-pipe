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

# Modifications by Juk Armstrong, November 2024:
#
#   Support for Logit Scaling:
#   - Added support for logit scaling via the `LOGIT_SCALE` parameter.
#   - This allows scaling logits by a constant factor before computing the loss for use with Cohere models.
#   - This also allows Focal Loss* to be implemented.
#   - See: https://arxiv.org/abs/1708.02002 (Appendix A/B)
#
#   Support for Label Smoothing:
#   - Added support for label smoothing via the `LABEL_SMOOTHING_LAMBDA` parameter.
#   - The smoothed labels are: y_smooth = (1 - λ) * y + λ * u, where u ~ Uniform(1 / VOCAB_SIZE)
#   - Loss = CE_loss(y_smooth, p)
#          = (1 - λ) * CE_loss(y, p) + λ * CE_loss(u, p)
#          = (1 - λ) * (-Σ y_i * log p_i) + λ * (-Σ u_i * log p_i)
#          = (1 - λ) * H(y, p) + λ * H(u, p)
#          = (1 - λ) * [H(y) + D_KL(y || p)] + λ * [H(u) + D_KL(u || p)]
#   - Since H(y) = 0 (because y is one-hot) and H(u) is constant, minimizing the loss is equivalent to minimizing:
#          (1 - λ) * D_KL(y || p) + λ * D_KL(u || p)
#   - The term D_KL(u || p) relates to the negative entropy of p:
#          D_KL(u || p) = log(VOCAB_SIZE) - H(p)
#     where log(VOCAB_SIZE) is constant with respect to p.
#   - Therefore, minimizing D_KL(u || p) is equivalent to maximizing H(p), the entropy of the predictions.
#   - This means label smoothing inherently adds an entropy regularization term λ * H(p),
#     encouraging higher entropy in predictions and reducing over-confidence.
#   - As a result, we do not need a separate entropy regularization function;
#     label smoothing already serves this purpose within the loss function.
#   - See: https://arxiv.org/abs/1701.06548 (Section 3.2)
#          https://arxiv.org/abs/1906.02629
#          https://arxiv.org/abs/1611.01838
#
#   Code Clean-Up and Documentation:
#   - Updated variable names for better readability and consistency with standard naming conventions.
#   - Added detailed comments explaining the existing/new mathematics and implementation details.
#
# NOTE: In the backward pass, the logits' gradients are stored directly into the `logits` tensor.
#       - This seems to be a deliberate design choice to save GPU memory...
#       - As a result, the original values of `logits` will be overwritten.

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
    LOGIT_SCALE: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    LABEL_SMOOTHING_LAMBDA: tl.constexpr,
):
    """
    Cross Entropy Loss = 1/n sum [ y_i log(p_i) ] where:
      p_i = exp(z_i) / sum_j exp(z_j)   # Softmax probability
      y_i = target label (one-hot or smoothed)

    For standard cross-entropy with one-hot targets:
      y_i = 1 if i == t (true label)
      y_i = 0 otherwise

    The loss simplifies to:
    CE = log(p_t)
       = [ z_t - logsumexp ]
       = logsumexp - z_t

    where:
      z_i: logit for class i
      z_t: logit corresponding to the true label
      logsumexp: log( sum_j exp(z_j) )

    Stability trick for computing logsumexp:
      logsumexp = c + log( sum_j exp(z_j - c) ), where c = max(z_j)

    For logit scaling:

      When logit scaling is applied, each logit z_i is scaled by s:
        z_i = s * z_i

    For label smoothing:

      With label smoothing, the targets y_i are modified:
        y_t = (1 - λ) for the true label
        y_i = λ / V   for all other labels i != t
      The loss becomes:
        CE = logsumexp - (1 - λ) * z_t + λ * log(V)
    """
    row_idx = tl.program_id(0)
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx
    labels_ptr    += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Create a mask to prevent out-of-bounds access for vocab sizes not divisible by BLOCK_SIZE
    mask = col_offsets < VOCAB_SIZE

    # Assuming label_idx ∈ [0, VOCAB_SIZE - 1] or -100 for ignore index
    label_idx = tl.load(labels_ptr).to(tl.int32)

    # Load logits z_i
    z = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    if DO_LOGIT_SCALING:
        # Apply logit scaling: z_i = s * z_i
        z = LOGIT_SCALE * z

    # Compute logsumexp = log( sum_j exp(z_j) ) using stability trick
    c = tl.max(z, 0)  # c = max(z_j)
    logsumexp = c + tl.log(tl.sum(tl.exp(z - c), 0))

    if label_idx != -100:
        # Load the logit corresponding to the true label: z_t
        z_t = tl.load(logits_ptr + label_idx).to(tl.float32)
        if DO_LOGIT_SCALING:
            # Apply logit scaling: z_t = s * z_t
            z_t = LOGIT_SCALE * z_t

        if DO_LABEL_SMOOTHING:
            # Compute loss with label smoothing:
            # CE = logsumexp - (1 - λ) * z_t + λ * log(V)
            loss = logsumexp - (1.0 - LABEL_SMOOTHING_LAMBDA) * z_t + LABEL_SMOOTHING_LAMBDA * tl.log(float(VOCAB_SIZE))
        else:
            # Standard cross-entropy loss: CE = logsumexp - z_t
            loss = logsumexp - z_t
    else:
        # If label is -100 (ignore index), set loss to 0
        loss = 0.0

    # Store results
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
    LOGIT_SCALE: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    LABEL_SMOOTHING_LAMBDA: tl.constexpr,
):
    """
    For large vocabularies, the vocabulary is divided into N_CHUNKS chunks.
    For example, a 256K vocabulary divided into 4 chunks of 65,536 each:

    |-65,536-| |-65,536-| |-65,536-| |-65,536-|
    |--------| |--------| |--------| |--------|
    |--------| |--------| |--------| |--------|

    Computing the full cross-entropy loss directly can be challenging due to memory constraints.
    Instead, we compute the log-sum-exp (logsumexp) over each chunk and then combine them to obtain the total logsumexp.

    TOTAL LOGSUMEXP COMPUTATION:

    The total logsumexp_total is computed as:
      logsumexp_total = log(sum_j exp(z_j)) = log( sum over all chunks [ sum_{i in chunk} exp(z_i) ] )

    We compute the logsumexp for each chunk:
      logsumexp_chunk = log( sum_{i in chunk} exp(z_i) )

    Then, we combine the per-chunk logsumexp values:
      logsumexp_total = log( sum over chunks [ exp(logsumexp_chunk) ] )

    IN THIS KERNEL:

      For each chunk, we compute the chunk's logsumexp and store it in logsumexp_ptr.
      When chunk_idx == 0, we compute the partial loss involving the true label's logit z_t.

    THE OVERALL LOSS IS COMPUTED OUTSIDE THIS KERNEL BY:

      Combining the per-chunk logsumexp values to get logsumexp_total.
      Adding the partial loss (computed here) to logsumexp_total to obtain the final loss:
        loss = logsumexp_total - z_t

      If label smoothing is enabled:
        loss = logsumexp_total - (1 - lambda) * z_t + lambda * log(V)

      where:
        lambda is the label smoothing coefficient (LABEL_SMOOTHING_LAMBDA)
        V is the vocabulary size (VOCAB_SIZE)

     For logit scaling:

      When logit scaling is applied, each logit z_i is scaled by s:
        z_i = s * z_i
      Similarly, the true label's logit z_t is scaled:
        z_t = s * z_t

    For label smoothing:

      The target labels y_i are modified:
        For the true label (i == t):   y_t = 1 - lambda
        For all other labels (i != t):  y_i = lambda / V
    """
    row_idx   = tl.program_id(0)
    chunk_idx = tl.program_id(1)

    # Adjust pointers to point to the correct row and chunk
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr    += row_idx

    # Compute column offsets for the current chunk
    col_offsets = chunk_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bounds access
    mask = col_offsets < VOCAB_SIZE
    # Load the label index for the current row
    label_idx = tl.load(labels_ptr).to(tl.int32)

    # Load logits z_i for the current chunk
    z = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if DO_LOGIT_SCALING:
        # Apply logit scaling: z_i = s * z_i
        z = LOGIT_SCALE * z

    # Compute logsumexp over the chunk using the stability trick
    c = tl.max(z, 0)  # c = max(z_i) over the chunk
    logsumexp_chunk = c + tl.log(tl.sum(tl.exp(z - c), 0))

    # Store the chunk's logsumexp
    tl.store(logsumexp_ptr, logsumexp_chunk)

    # For the first chunk, compute the partial loss involving z_t
    if chunk_idx == 0:
        if label_idx != -100:
            # Load the logit corresponding to the true label: z_t
            z_t = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING:
                # Apply logit scaling: z_t = s * z_t
                z_t = LOGIT_SCALE * z_t

            if DO_LABEL_SMOOTHING:
                # Compute the partial loss with label smoothing:
                # loss = - (1 - lambda) * z_t + lambda * log(V)
                loss = - (1.0 - LABEL_SMOOTHING_LAMBDA) * z_t + LABEL_SMOOTHING_LAMBDA * tl.log(float(VOCAB_SIZE))
            else:
                # Standard cross-entropy loss: loss = - z_t
                loss = - z_t
        else:
            # If label is -100 (ignore index), set loss to 0
            loss = 0.0

        # Store the partial loss
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
    LOGIT_SCALE: tl.constexpr,
    DO_LABEL_SMOOTHING: tl.constexpr,
    LABEL_SMOOTHING_LAMBDA: tl.constexpr,
):
    """
    Compute gradients of the cross-entropy loss with respect to logits z_i.

    The gradient dL/dz_i is given by:
      dL/dz_i = p_i - y_i

    where:
      p_i = exp(z_i) / sum_j exp(z_j)  # Softmax probability
      y_i = target label

    For logit scaling:

      With logit scaling (z_i = s * z_i), the gradient becomes:
        dL/dz_i = s * (p_i - y_i)

    For label smoothing:

      y_t = (1 - λ) for the true label
      y_i = λ / V   for all other labels i != t
    """
    row_idx   = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr  += row_idx * dloss_row_stride

    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a mask to prevent out-of-bounds access for vocab sizes not divisible by BLOCK_SIZE
    mask = col_offsets < VOCAB_SIZE

    # Assuming label_idx ∈ [0, VOCAB_SIZE - 1] or -100 for ignore index
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        # Load upstream gradient: dL/dloss
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    # Load logits z_i
    z = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    if DO_LOGIT_SCALING:
        # Apply logit scaling: z_i = s * z_i
        z = LOGIT_SCALE * z

    # Load logsumexp = log( sum_j exp(z_j) )
    logsumexp = tl.load(logsumexp_ptr + row_idx)

    # Compute softmax probabilities: p_i = exp(z_i - logsumexp)
    p = tl.exp(z - logsumexp)

    if DO_LABEL_SMOOTHING:
        # Compute target labels y_i with label smoothing
        # y_t = (1 - λ), y_i = λ / V for i != t
        y_i = LABEL_SMOOTHING_LAMBDA / float(VOCAB_SIZE)
        y = tl.full_like(p, y_i)
        y = tl.where(col_offsets == label_idx, 1.0 - LABEL_SMOOTHING_LAMBDA, y)
    else:
        # Standard one-hot target labels
        y = tl.where(col_offsets == label_idx, 1.0, 0.0)

    # Compute gradient: dL/dz_i = p_i - y_i
    d_logits = p - y

    if DO_LOGIT_SCALING:
        # Account for logit scaling in gradient: dL/dz_i = s * (p_i - y_i)
        d_logits = LOGIT_SCALE * d_logits

    # Multiply with upstream gradient dL/dloss
    grad = dloss * d_logits

    # Store gradient w.r.t logits
    # - This seems to be a deliberate design choice to save GPU memory...
    # - As a result, the original values of `logits` will be overwritten.
    tl.store(logits_ptr + col_offsets, grad, mask=mask)


MAX_FUSED_SIZE = 65536 # 2**16

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_scale = 1.0, label_smoothing_lambda = 0.0):
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
                DO_LOGIT_SCALING = (logit_scale != 1.0),
                LOGIT_SCALE = logit_scale,
                DO_LABEL_SMOOTHING = (label_smoothing_lambda != 0.0),
                LABEL_SMOOTHING_LAMBDA = label_smoothing_lambda,
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
                DO_LOGIT_SCALING = (logit_scale != 1.0),
                LOGIT_SCALE = logit_scale,
                DO_LABEL_SMOOTHING = (label_smoothing_lambda != 0.0),
                LABEL_SMOOTHING_LAMBDA = label_smoothing_lambda,
                num_warps  = 32,
            )
            # logsumexp(chunked_logsumexp) - x
            # Do the -x separately
            logsumexp = torch.logsumexp(logsumexp, dim = 1) # Row sum
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0) # Don't forget to mask padding out!

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.logit_scale = logit_scale
        ctx.label_smoothing_lambda = label_smoothing_lambda
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
            DO_LOGIT_SCALING = (ctx.logit_scale != 1.0),
            LOGIT_SCALE = ctx.logit_scale,
            DO_LABEL_SMOOTHING = (ctx.label_smoothing_lambda != 0.0),
            LABEL_SMOOTHING_LAMBDA = ctx.label_smoothing_lambda,
            num_warps  = 8,
        )
        return logits, None, None, None


def fast_cross_entropy_loss(logits, labels, logit_scale=1.0, label_smoothing_lambda = 0.0):
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
        logit_scale,
        label_smoothing_lambda,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
