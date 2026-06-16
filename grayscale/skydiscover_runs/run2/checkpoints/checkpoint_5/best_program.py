# EVOLVE-BLOCK-START
"""
Initial Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel(
    rgb_ptr, out_ptr,
    n_pixels,
    BLOCK_SIZE: tl.constexpr,
):
    """RGB->gray with fully coalesced contiguous loads.

    Each program loads a contiguous block of 3*BLOCK_SIZE source elements
    (the interleaved RGB stream) using three coalesced loads at consecutive
    base offsets, then deinterleaves via strided indices to compute Y.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    base = offsets * 3
    r = tl.load(rgb_ptr + base, mask=mask, other=0.0)
    g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    """RGB->grayscale using flat contiguous loads for maximal coalescing.

    Uses a larger BLOCK_SIZE and tuned warp/stage counts to saturate
    H100 memory bandwidth for the bandwidth-bound conversion.
    """
    rgb, output = data
    H, W, C = rgb.shape
    rgb = rgb.contiguous()
    n_pixels = H * W
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return output
# EVOLVE-BLOCK-END
