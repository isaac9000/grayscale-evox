# EVOLVE-BLOCK-START
"""
Grayscale via inline CUDA kernel: 4-pixel float4 with non-temporal stores.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Best kernel (#10) + non-temporal float4 store via __stcg() for output:
- Output buffer is write-only; non-temporal store bypasses L1/L2 cache
  pollution, routing writes directly to L2 / HBM for large sizes.
- __stcg = "store cache global" — bypasses L1, goes to L2
- Keeps: __launch_bounds__(256,4), float4 loads, __ldg reads.
"""

import torch
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <cuda_runtime.h>

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

        // Non-temporal store: bypass L1, reduce cache pollution for write-only output
        __stcg(reinterpret_cast<float4*>(out) + idx / 4, result);
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

torch::Tensor grayscale_cuda(torch::Tensor rgb, torch::Tensor output) {
    int n_pixels = output.numel();
    const int threads = 256;
    int blocks = (n_pixels / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
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
            name="grayscale_inline_v10",
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
