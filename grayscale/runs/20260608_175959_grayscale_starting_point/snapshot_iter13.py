# EVOLVE-BLOCK-START
"""
Grayscale via inline CUDA kernel: 4-pixel float4 with __launch_bounds__.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Based on best kernel (#5). Additions:
- __launch_bounds__(256, 4): gives compiler explicit occupancy hint for
  better register allocation and reduced spill on H100.
- Eliminated rgb.contiguous() (input is guaranteed contiguous) and .view(-1)
  (pass data pointer directly) to reduce Python-side per-call overhead.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void __launch_bounds__(256, 4) grayscale_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int n_pixels
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n_pixels) {
        const float4* rgb4 = reinterpret_cast<const float4*>(rgb + idx * 3);
        float4 v0 = __ldg(rgb4);
        float4 v1 = __ldg(rgb4 + 1);
        float4 v2 = __ldg(rgb4 + 2);

        float4 result;
        result.x = 0.2989f * v0.x + 0.5870f * v0.y + 0.1140f * v0.z;
        result.y = 0.2989f * v0.w + 0.5870f * v1.x + 0.1140f * v1.y;
        result.z = 0.2989f * v1.z + 0.5870f * v1.w + 0.1140f * v2.x;
        result.w = 0.2989f * v2.y + 0.5870f * v2.z + 0.1140f * v2.w;

        reinterpret_cast<float4*>(out)[idx / 4] = result;
    } else {
        for (int i = idx; i < n_pixels; i++) {
            int base = i * 3;
            float r = __ldg(rgb + base);
            float g = __ldg(rgb + base + 1);
            float b = __ldg(rgb + base + 2);
            out[i] = 0.2989f * r + 0.5870f * g + 0.1140f * b;
        }
    }
}

// Accept raw integer pointers to avoid ATen tensor metadata unpacking overhead
void grayscale_cuda(int64_t rgb_ptr, int64_t out_ptr, int n_pixels) {
    const int threads = 256;
    int blocks = (n_pixels / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    grayscale_kernel<<<blocks, threads>>>(
        reinterpret_cast<const float*>(rgb_ptr),
        reinterpret_cast<float*>(out_ptr),
        n_pixels
    );
}
"""

_cpp_src = r"""
void grayscale_cuda(int64_t rgb_ptr, int64_t out_ptr, int n_pixels);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="grayscale_inline_v6",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["grayscale_cuda"],
            verbose=False,
        )
    return _module

def custom_kernel(data):
    rgb, output = data
    mod = _get_module()
    # Pass raw integer pointers — avoids ATen tensor metadata unpacking
    mod.grayscale_cuda(rgb.data_ptr(), output.data_ptr(), output.numel())
    return output
# EVOLVE-BLOCK-END
