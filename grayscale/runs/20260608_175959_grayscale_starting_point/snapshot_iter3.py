# EVOLVE-BLOCK-START
"""
Grayscale via pure PyTorch ops.
Y = 0.2989 R + 0.5870 G + 0.1140 B

Uses rgb @ weights (matrix-vector product) on the contiguous (H*W, 3) view.
Avoids Triton JIT warm-up and kernel launch overhead at small sizes.
cuBLAS/cublasLt handles the reduction efficiently.
"""

import torch

_weights = None

def custom_kernel(data):
    global _weights
    rgb, output = data
    H, W, C = rgb.shape
    # Pre-allocate weights tensor on the same device (cached)
    if _weights is None or _weights.device != rgb.device:
        _weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                dtype=torch.float32, device=rgb.device)
    # rgb is (H, W, 3) contiguous float32; reshape to (H*W, 3)
    # matrix-vector multiply: (H*W, 3) @ (3,) -> (H*W,)
    gray = rgb.view(-1, 3) @ _weights
    output.copy_(gray.view(H, W))
    return output
# EVOLVE-BLOCK-END
