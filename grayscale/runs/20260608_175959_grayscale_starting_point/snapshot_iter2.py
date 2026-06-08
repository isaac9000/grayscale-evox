# EVOLVE-BLOCK-START
"""
Optimized Grayscale submission with Triton kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Changes from baseline:
- Treat input as flat (H*W*3,) array to eliminate // and % operations
- Load R, G, B using flat base pointer: base = pixel_idx * 3
- Use autotune over BLOCK_SIZE to handle both small and large images
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_pixels'],
)
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

    # Flat indexing: pixel i has R at 3i, G at 3i+1, B at 3i+2
    base = offsets * 3
    r = tl.load(rgb_ptr + base,     mask=mask, other=0.0)
    g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    tl.store(out_ptr + offsets, gray, mask=mask)


def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb_flat = rgb.contiguous().view(-1)
    n_pixels = H * W
    grid = lambda meta: (triton.cdiv(n_pixels, meta['BLOCK_SIZE']),)
    grayscale_kernel[grid](
        rgb_flat, output.view(-1),
        n_pixels,
    )
    return output
# EVOLVE-BLOCK-END
