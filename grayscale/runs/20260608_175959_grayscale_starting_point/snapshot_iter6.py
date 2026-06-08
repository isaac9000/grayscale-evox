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

// Each thread processes 8 pixels using float4 vectorized loads.
// 8 pixels = 24 floats = 6 float4 loads (RGBRGB... layout).
// Issue all 6 loads first (maximizes MLP), then compute 8 grayscale values,
// then store as 2x float4. This hides memory latency via ILP.
__global__ void grayscale_kernel(
    const float* __restrict__ rgb,
    float* __restrict__ out,
    int n_pixels
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    if (idx + 7 < n_pixels) {
        // Full 8-pixel path: 6x float4 loads (issue all first for MLP)
        const float4* rgb4 = reinterpret_cast<const float4*>(rgb + idx * 3);
        // Issue all loads before any compute
        float4 v0 = __ldg(rgb4);     // R0 G0 B0 R1
        float4 v1 = __ldg(rgb4 + 1); // G1 B1 R2 G2
        float4 v2 = __ldg(rgb4 + 2); // B2 R3 G3 B3
        float4 v3 = __ldg(rgb4 + 3); // R4 G4 B4 R5
        float4 v4 = __ldg(rgb4 + 4); // G5 B5 R6 G6
        float4 v5 = __ldg(rgb4 + 5); // B6 R7 G7 B7

        // Compute 8 grayscale values
        float4 res0, res1;
        res0.x = 0.2989f * v0.x + 0.5870f * v0.y + 0.1140f * v0.z; // pixel 0
        res0.y = 0.2989f * v0.w + 0.5870f * v1.x + 0.1140f * v1.y; // pixel 1
        res0.z = 0.2989f * v1.z + 0.5870f * v1.w + 0.1140f * v2.x; // pixel 2
        res0.w = 0.2989f * v2.y + 0.5870f * v2.z + 0.1140f * v2.w; // pixel 3
        res1.x = 0.2989f * v3.x + 0.5870f * v3.y + 0.1140f * v3.z; // pixel 4
        res1.y = 0.2989f * v3.w + 0.5870f * v4.x + 0.1140f * v4.y; // pixel 5
        res1.z = 0.2989f * v4.z + 0.5870f * v4.w + 0.1140f * v5.x; // pixel 6
        res1.w = 0.2989f * v5.y + 0.5870f * v5.z + 0.1140f * v5.w; // pixel 7

        float4* out4 = reinterpret_cast<float4*>(out);
        out4[idx / 8 * 2]     = res0;
        out4[idx / 8 * 2 + 1] = res1;
    } else {
        // Tail: handle remaining pixels scalarly
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
    int n_pixels = rgb.numel() / 3;
    const int threads = 256;
    // Each thread handles 8 pixels
    int blocks = (n_pixels / 8 + threads - 1) / threads;
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
