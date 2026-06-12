# EVOLVE-BLOCK-START
import torch
import triton
import triton.language as tl


@triton.jit
def grayscale_kernel_hwc(rgb_ptr, out_ptr, n_pixels, BLOCK_SIZE: tl.constexpr):
    """HWC contiguous float32: pixel i has R@3i, G@3i+1, B@3i+2. Stride-3 loads, coalesced output. num_warps=8 for H100."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_pixels
    base = offsets * 3
    r = tl.load(rgb_ptr + base,     mask=mask, other=0.0)
    g = tl.load(rgb_ptr + base + 1, mask=mask, other=0.0)
    b = tl.load(rgb_ptr + base + 2, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, r * 0.2989 + g * 0.5870 + b * 0.1140, mask=mask)


def custom_kernel(data):
    """Grayscale Y=0.2989R+0.5870G+0.1140B via Triton HWC kernel, BLOCK_SIZE=1024, num_warps=8, num_stages=2."""
    rgb, output = data
    n_pixels = rgb.shape[0] * rgb.shape[1]
    BLOCK_SIZE = 1024
    grayscale_kernel_hwc[(triton.cdiv(n_pixels, BLOCK_SIZE),)](
        rgb.contiguous(), output, n_pixels,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2,
    )
    return output
# EVOLVE-BLOCK-END
