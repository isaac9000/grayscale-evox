# EVOLVE-BLOCK-START
"""
Grayscale via inline CUDA kernel using float3 vector loads.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Each thread loads one pixel's RGB as 3 consecutive floats, computes
weighted sum, writes result. Uses 256 threads/block for high occupancy.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>

__global__ void grayscale_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int n_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;
    int base = idx * 3;
    float r = rgb[base];
    float g = rgb[base + 1];
    float b = rgb[base + 2];
    out[idx] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
}

torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output) {
    int n_pixels = rgb.numel() / 3;
    const int threads = 256;
    int blocks = (n_pixels + threads - 1) / threads;
    grayscale_kernel<<<blocks, threads>>>(
        rgb.data_ptr<float>(),
        output.data_ptr<float>(),
        n_pixels
    );
    return output;
}
"""

_cpp_src = r"""
torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="grayscale_inline",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["grayscale_cuda"],
            verbose=False,
        )
    return _module

def custom_kernel(data):
    rgb, output = data
    H, W, C = rgb.shape
    assert C == 3
    rgb_c = rgb.contiguous()
    mod = _get_module()
    mod.grayscale_cuda(rgb_c.view(-1), output.view(-1))
    return output
# EVOLVE-BLOCK-END
