# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py
import math
import time

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from flash_attn import flash_attn_with_kvcache

if torch.cuda.is_available():
    TEST_DEVICE = "cuda"
    # from flash_attn import flash_attn_with_kvcache
elif hasattr(torch, "npu") and torch.npu.is_available():
    TEST_DEVICE = "npu"
    # import torch_npu
    # from flash_attn_npu import flash_attn_with_kvcache
else:
    raise RuntimeError("No supported device found (CUDA/NPU)")

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    key_padding_mask=None,
    device=None
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = seqlen_q
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False
):
    """
    Device Naive reference implementation of attention, used for testing precision of FlashAttention.
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        key_padding_mask: (batch_size, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    # set seed
    torch.random.manual_seed(0)
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            key_padding_mask,
            q.device
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def _generate_block_kvcache(
    seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
):
    num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
    k_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    v_cache_paged = torch.randn(
        num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
    )
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        k_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    v_cache = rearrange(
        v_cache_paged[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks


def test_flash_attn_kvcache(
    seqlen_q,
    seqlen_k,
    d,
    paged_kv_block_size,
    causal,
    mha_type,
    num_splits,
    dtype,
):
    """
    Flash-attn-npu/gpu implementation of attention with kv cache, compared against a naive PyTorch implementation.
    """
    
    device = TEST_DEVICE
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    batch_size_cache = batch_size
    nheads = 6
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1)
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k, v = None, None
    (
        k_cache,
        v_cache,
        block_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = _generate_block_kvcache(
        seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype
    )
    cache_seqlens = torch.randint(
        1,
        # If we don't use seqlen_q in the case of causal and rotary, cos/sin won't be long enough
        seqlen_k - seqlen_q + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    key_padding_mask = arange < cache_seqlens_expanded
    q_ro = q
    # k_cache[:, 64:] = -1
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    
    # NPU/GPU implementation with kv cache
    out = flash_attn_with_kvcache(
        q,
        k_cache_paged,
        v_cache_paged,
        k,
        v,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=causal,
        window_size=window_size,
        num_splits=num_splits,
    )

    # Device Naive reference implementation with kv cache
    out_ref, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        key_padding_mask,
        causal=causal,
        window_size=window_size
    )
    
    # pytorch precision comparison
    out_pt, _ = attention_ref(
        q_ro,
        k_cache_rep,
        v_cache_rep,
        key_padding_mask,
        causal=causal,
        window_size=window_size,
        upcast=False,
        reorder_ops=True
    )    
    
    #? Maybe for better presentation?
    mult = 3
    num_elements = out.numel()

    print("\n" + "="*70)
    print(f"{'指标':<12} {'朴素实现':<10} {'Flash Attention':<19} {'是否在误差容忍内':<5}")
    print("="*70)

    max_diff = (out - out_ref).abs().max().item()
    max_diff_pt = (out_pt - out_ref).abs().max().item()
    max_pass = max_diff <= mult * max_diff_pt + 1e-5

    # Below are custom metrics that we think better capture the numerical error 
    # of FlashAttention compared to a PyTorch implementation, rather than just looking at max error.
    mean_diff = (out - out_ref).abs().mean().item()
    mean_diff_pt = (out_pt - out_ref).abs().mean().item()
    mean_pass = mean_diff <= mult * mean_diff_pt + 1e-8

    l1_diff = torch.norm(out - out_ref, p=1).item()
    l1_diff_pt = torch.norm(out_pt - out_ref, p=1).item()
    l1_per_element = l1_diff / num_elements
    l1_pt_per_element = l1_diff_pt / num_elements
    l1_pass = l1_per_element <= mult * l1_pt_per_element + 1e-5

    l2_diff = torch.norm(out - out_ref, p=2).item()
    l2_diff_pt = torch.norm(out_pt - out_ref, p=2).item()
    l2_per_element = l2_diff / math.sqrt(num_elements)
    l2_pt_per_element = l2_diff_pt / math.sqrt(num_elements)
    l2_pass = l2_per_element <= mult * l2_pt_per_element + 1e-5

    cos_diff = torch.clamp(
        torch.cosine_similarity(out.flatten(), out_ref.flatten(), dim=0),
        min=-1.0, max=1.0,
    ).item()
    cos_diff_pt = torch.clamp(
        torch.cosine_similarity(out_pt.flatten(), out_ref.flatten(), dim=0),
        min=-1.0, max=1.0,
    ).item()
    cos_pass = cos_diff <= cos_diff_pt + 1e-5

    print(f"{'最大误差':<11} {max_diff_pt:<17.5f} {max_diff:<20.5f} {str(max_pass):<5}")
    print(f"{'平均误差':<11} {mean_diff_pt:<17.5f} {mean_diff:<20.5f} {str(mean_pass):<5}")
    print(f"{'L1 范数':<13} {l1_diff_pt:<17.5f} {l1_diff:<20.5f} {str(l1_pass):<5}")
    print(f"{'L2 范数':<13} {l2_diff_pt:<17.5f} {l2_diff:<20.5f} {str(l2_pass):<5}")
    print(f"{'余弦相似度':<10} {cos_diff_pt:<17.5f} {cos_diff:<20.5f} {str(cos_pass):<5}")
    print("="*70 + "\n")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5

if __name__ == "__main__":
    test_flash_attn_kvcache(
        seqlen_q=16,
        seqlen_k=1024,
        d=128,
        paged_kv_block_size=256,
        causal=True,
        mha_type="mha",
        num_splits=1,
        dtype=torch.float16,
    )