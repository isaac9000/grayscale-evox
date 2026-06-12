# EVOLVE-BLOCK-START
"""
Optimized Grayscale kernel for H100.
Y = 0.2989 R + 0.5870 G + 0.1140 B
Uses vectorized loads in HWC contiguous layout: pixel i has R at 3*i, G at 3*i+1, B at 3*i+2.
Avoids integer division (no h_idx/w_idx), minimal arithmetic overhead.
BLOCK_SIZE=1024 tuned for H100 warp occupancy and L2 cache efficiency.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel_hwc(
    rgb_ptr, out_ptr,
    n_pixels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast path kernel for contiguous HWC float32 layout on H100.
    Loads R, G, B with stride-3 access pattern (no div/mod needed).
    Each block processes BLOCK_SIZE pixels with coalesced output writes.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    base = offsets * 3
    r = tl.load(rgb_ptr + base,     mask=mask, other=0.0)
    g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)

    gray = r * 0.2989 + g * 0.5870 + b * 0.1140

    tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    """Launch optimized grayscale kernel using HWC fast path on H100."""
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    n_pixels = H * W
    BLOCK_SIZE = 1024

    rgb_c = rgb.contiguous()
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel_hwc[grid](
        rgb_c, output,
        n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END
