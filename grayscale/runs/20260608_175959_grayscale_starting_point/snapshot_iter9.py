# EVOLVE-BLOCK-START
"""
Grayscale via torch.compile with reduce-overhead mode.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Uses torch.compile(mode="reduce-overhead") which internally captures CUDA graphs
without requiring explicit static buffer management. The inductor backend fuses
the element-wise ops into a single kernel and eliminates Python dispatch overhead
on repeated calls of the same shape.
"""

import torch

def _grayscale_fn(rgb):
    return rgb[..., 0] * 0.2989 + rgb[..., 1] * 0.5870 + rgb[..., 2] * 0.1140

_compiled_fn = torch.compile(_grayscale_fn, mode="reduce-overhead")

def custom_kernel(data):
    rgb, output = data
    result = _compiled_fn(rgb)
    output.copy_(result)
    return output
# EVOLVE-BLOCK-END
