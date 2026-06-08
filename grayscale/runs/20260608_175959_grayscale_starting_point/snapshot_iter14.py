# EVOLVE-BLOCK-START
"""
Grayscale via persistent grid-stride CUDA kernel.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Fixed grid of 132 SMs × 4 blocks × 256 threads = 135,168 total threads.
Each thread loops over its assigned pixels using a grid stride, ensuring
all H100 SMs are always fully saturated regardless of image size.
Inner loop body: same 4-pixel float4 loads as best kernel (#10).
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>

// Persistent grid-stride kernel: fixed grid saturates all 132 H100 SMs.
// Each thread processes chunks of 4 pixels, striding by total_threads*4.
__global__ void __launch_bounds__(256, 4) grayscale_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int n_pixels
) {
    int total_threads = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes groups of 4 pixels, striding by total_threads
    for (int pixel_group = tid; pixel_group * 4 + 3 < n_pixels; pixel_group += total_threads) {
        int idx = pixel_group * 4;
        const float4* rgb4 = reinterpret_cast<const float4*>(rgb + idx * 3);
        float4 v0 = __ldg(rgb4);
        float4 v1 = __ldg(rgb4 + 1);
        float4 v2 = __ldg(rgb4 + 2);

        float4 result;
        result.x = 0.2989f * v0.x + 0.5870f * v0.y + 0.1140f * v0.z;
        result.y = 0.2989f * v0.w + 0.5870f * v1.x + 0.1140f * v1.y;
        result.z = 0.2989f * v1.z + 0.5870f * v1.w + 0.1140f * v2.x;
        result.w = 0.2989f * v2.y + 0.5870f * v2.z + 0.1140f * v2.w;

        reinterpret_cast<float4*>(out)[pixel_group] = result;
    }

    // Scalar tail for remaining pixels
    int tail_start = (n_pixels / 4) * 4;
    for (int i = tail_start + tid; i < n_pixels; i += total_threads) {
        int base = i * 3;
        float r = __ldg(rgb + base);
        float g = __ldg(rgb + base + 1);
        float b = __ldg(rgb + base + 2);
        out[i] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
    }
}

torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output) {
    int n_pixels = output.numel();
    const int threads = 256;
    // Fixed grid: 132 SMs × 4 blocks each = 528 blocks total
    const int blocks = 132 * 4;
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
            name="grayscale_inline_v7",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["grayscale_cuda"],
            verbose=False,
        )
    return _module

def custom_kernel(data):
    rgb, output = data
    mod = _get_module()
    mod.grayscale_cuda(rgb, output)
    return output
# EVOLVE-BLOCK-END
