"""INT8 quantization baseline (per-channel symmetric), CPU-portable.

Mirrors bitsandbytes' LLM.int8() storage model closely enough for fair
parameter/accuracy comparison without requiring CUDA. When CUDA +
bitsandbytes are available, `Int8Linear.use_bitsandbytes` can be flipped by
the benchmark harness.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    """Linear layer with weights stored as int8 + per-output-channel scale."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        W = linear.weight.data
        scale = W.abs().amax(dim=1, keepdim=True) / 127.0
        scale = torch.clamp(scale, min=1e-8)
        q = torch.round(W / scale).to(torch.int8)
        self.register_buffer("weight_int8", q)
        self.register_buffer("scale", scale)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight_int8.to(x.dtype) * self.scale
        return nn.functional.linear(x, W, self.bias)

    def effective_param_bytes(self) -> int:
        """int8 weights = 1 byte/param; scale + bias in fp32."""
        n = self.weight_int8.numel() + 4 * self.scale.numel()
        if self.bias is not None:
            n += 4 * self.bias.numel()
        return n


def quantize_model_int8(model: nn.Module, min_features: int = 64) -> nn.Module:
    """Replace eligible nn.Linear layers with Int8Linear (in place)."""
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_features:
                setattr(module, name, Int8Linear(child))
    return model
