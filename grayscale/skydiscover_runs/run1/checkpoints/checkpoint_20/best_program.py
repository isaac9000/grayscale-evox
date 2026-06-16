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
    EVEN: tl.constexpr,
):
    """RGB->gray with coalesced de-interleaved loads + fma.

    Each program processes BLOCK_SIZE pixels via three coalesced strided
    loads at consecutive base offsets (no div/mod). On the EVEN fast-path
    (n_pixels divisible by BLOCK_SIZE, always true for square power-of-two
    inputs) all masking is dropped to cut transaction/compare overhead.
    fma fuses the weight chain into two fused multiply-adds.
    """
    WR: tl.constexpr = 0.2989
    WG: tl.constexpr = 0.5870
    WB: tl.constexpr = 0.1140
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    base = offsets * 3
    if EVEN:
        r = tl.load(rgb_ptr + base)
        g = tl.load(rgb_ptr + base + 1)
        b = tl.load(rgb_ptr + base + 2)
        gray = tl.fma(WR, r, tl.fma(WG, g, WB * b))
        tl.store(out_ptr + offsets, gray)
    else:
        mask = offsets < n_pixels
        r = tl.load(rgb_ptr + base, mask=mask, other=0.0)
        g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
        b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)
        gray = tl.fma(WR, r, tl.fma(WG, g, WB * b))
        tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    """RGB->grayscale via flat contiguous strided loads (no div/mod).

    Uses BLOCK_SIZE=4096 with 8 warps / 4 stages to saturate H100 HBM
    bandwidth, and drops masking on the EVEN fast-path when n_pixels
    divides the block size evenly (true for square power-of-two inputs).
    """
    rgb, output = data
    H, W, C = rgb.shape
    rgb = rgb.contiguous()
    n_pixels = H * W
    BLOCK_SIZE = 4096
    even = (n_pixels % BLOCK_SIZE) == 0
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output, n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
        EVEN=even,
        num_warps=8,
        num_stages=4,
    )
    return output
# EVOLVE-BLOCK-END
