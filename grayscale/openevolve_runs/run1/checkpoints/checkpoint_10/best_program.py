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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels

    # Contiguous HxWx3: pixel i -> R at 3i, G at 3i+1, B at 3i+2
    base = offsets * 3

    r = tl.load(rgb_ptr + base,     mask=mask)
    g = tl.load(rgb_ptr + base + 1, mask=mask)
    b = tl.load(rgb_ptr + base + 2, mask=mask)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb = rgb.contiguous()
    n_pixels = H * W
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    grayscale_kernel[grid](
        rgb, output,
        n_pixels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
# EVOLVE-BLOCK-END
