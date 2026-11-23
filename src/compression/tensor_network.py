"""Quantum-inspired tensor network compression for VLAM weight matrices.

Implements Matrix Product State (MPS / tensor-train) and Tree Tensor Network
(TTNS) decompositions of dense linear layers, following the CompactifAI
approach (Tomut et al., 2024, arXiv:2401.14109).

A dense weight W in R^{out x in} is reshaped into a higher-order tensor and
factorized into a chain (MPS) or binary tree (TTNS) of small cores whose
contraction approximates W. Bond dimension chi controls the
compression/accuracy trade-off.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _factorize(n: int, num_factors: int) -> List[int]:
    """Split integer n into `num_factors` roughly-balanced factors (product == n)."""
    factors: List[int] = []
    remaining = n
    for i in range(num_factors, 1, -1):
        target = round(remaining ** (1.0 / i))
        # find divisor of `remaining` closest to target
        best = 1
        for d in range(1, remaining + 1):
            if remaining % d == 0 and abs(d - target) < abs(best - target):
                best = d
            if d > target * 2 and best != 1:
                break
        factors.append(best)
        remaining //= best
    factors.append(remaining)
    assert math.prod(factors) == n, (factors, n)
    return factors


@dataclass
class CompressionStats:
    original_params: int
    compressed_params: int
    bond_dimension: int
    relative_error: float

    @property
    def compression_ratio(self) -> float:
        return self.original_params / max(self.compressed_params, 1)


class MPSLinear(nn.Module):
    """Linear layer whose weight is stored as a tensor-train (MPS).

    The dense weight (out x in) is reshaped to shape
    (o1, i1, o2, i2, ..., ok, ik) and decomposed by sequential truncated SVD
    into cores G_j of shape (chi_{j-1}, o_j * i_j, chi_j).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bond_dimension: int = 32,
        num_cores: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bond_dimension = bond_dimension
        self.in_factors = _factorize(in_features, num_cores)
        self.out_factors = _factorize(out_features, num_cores)
        self.cores = nn.ParameterList()
        chi_prev = 1
        for j in range(num_cores):
            phys = self.out_factors[j] * self.in_factors[j]
            chi_next = 1 if j == num_cores - 1 else bond_dimension
            core = torch.randn(chi_prev, phys, chi_next) * (1.0 / math.sqrt(phys * chi_prev))
            self.cores.append(nn.Parameter(core))
            chi_prev = chi_next
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._stats: Optional[CompressionStats] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bond_dimension: int = 32,
        num_cores: int = 3,
    ) -> "MPSLinear":
        """Decompose an existing nn.Linear into MPS form via truncated TT-SVD."""
        mps = cls(
            linear.in_features,
            linear.out_features,
            bond_dimension=bond_dimension,
            num_cores=num_cores,
            bias=linear.bias is not None,
        )
        W = linear.weight.data.detach().clone()  # (out, in)
        of, inf_ = mps.out_factors, mps.in_factors
        k = num_cores
        # reshape to (o1..ok, i1..ik) then permute to interleave (o1,i1,o2,i2,...)
        T = W.reshape(*of, *inf_)
        perm = []
        for j in range(k):
            perm.extend([j, k + j])
        T = T.permute(*perm).contiguous()
        # TT-SVD
        cores: List[torch.Tensor] = []
        chi_prev = 1
        rest = T.reshape(chi_prev, -1)
        for j in range(k - 1):
            phys = of[j] * inf_[j]
            mat = rest.reshape(chi_prev * phys, -1)
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            chi = min(bond_dimension, S.shape[0])
            U, S, Vh = U[:, :chi], S[:chi], Vh[:chi]
            cores.append(U.reshape(chi_prev, phys, chi))
            rest = (torch.diag(S) @ Vh)
            chi_prev = chi
        cores.append(rest.reshape(chi_prev, of[-1] * inf_[-1], 1))
        # actual ranks may be smaller than requested bond dimension
        mps.cores = nn.ParameterList(nn.Parameter(c.contiguous()) for c in cores)
        if linear.bias is not None:
            assert mps.bias is not None
            mps.bias.data.copy_(linear.bias.data)
        # stats
        W_hat = mps.reconstruct_weight()
        rel_err = (torch.norm(W - W_hat) / torch.norm(W)).item()
        orig = linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0)
        comp = sum(p.numel() for p in mps.parameters())
        mps._stats = CompressionStats(orig, comp, bond_dimension, rel_err)
        logger.info(
            "MPS decomposition: %dx%d -> %.2fx compression, rel_err=%.4f",
            linear.out_features, linear.in_features,
            mps._stats.compression_ratio, rel_err,
        )
        return mps

    # ------------------------------------------------------------------ #
    def reconstruct_weight(self) -> torch.Tensor:
        """Contract cores back into the dense (out, in) weight matrix."""
        k = len(self.cores)
        res = self.cores[0]  # (1, p0, chi)
        for j in range(1, k):
            res = torch.einsum("apb,bqc->apqc", res, self.cores[j])
            res = res.reshape(res.shape[0], -1, res.shape[-1])
        res = res.reshape(*[d for j in range(k) for d in (self.out_factors[j], self.in_factors[j])])
        # inverse permute: currently (o1,i1,o2,i2,...) -> want (o1..ok, i1..ik)
        inv = [2 * j for j in range(k)] + [2 * j + 1 for j in range(k)]
        res = res.permute(*inv).contiguous()
        return res.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Contract input against cores directly — never materializes dense W.

        Maintains T of shape (B, O, i_j, R, c): O = flattened output dims
        produced so far, i_j = current input factor, R = remaining input
        factors, c = current bond. Each step contracts i_j and the bond with
        core_j (c, o_j, i_j, c'), growing O and shrinking R.
        Cost O(B * k * chi^2 * phys) vs O(B * in * out) dense.
        """
        k = len(self.cores)
        lead = x.shape[:-1]
        B = int(torch.tensor(lead).prod().item()) if lead else 1
        h = x.reshape(B, self.in_features)
        rest = self.in_features // self.in_factors[0]
        T = h.reshape(B, 1, self.in_factors[0], rest, 1)
        for j in range(k):
            c_in, _, c_out = self.cores[j].shape
            core = self.cores[j].reshape(c_in, self.out_factors[j],
                                         self.in_factors[j], c_out)
            # (B,O,i,R,c) x (c,p,i,d) -> (B,O,p,R,d)
            T = torch.einsum("boirc,cpid->boprd", T, core)
            Bo, O, p, R, d = T.shape
            if j + 1 < k:
                nxt = self.in_factors[j + 1]
                T = T.reshape(B, O * p, nxt, R // nxt, d)
            else:
                T = T.reshape(B, O * p, 1, R, d)
        out = T.reshape(B, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.out_features)

    def get_compression_ratio(self) -> float:
        orig = self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0
        )
        return orig / sum(p.numel() for p in self.parameters())

    @property
    def stats(self) -> Optional[CompressionStats]:
        return self._stats


# Backwards-compatible alias used in docs/demo
MPSCompressor = MPSLinear


class TTNSLinear(nn.Module):
    """Two-level tree tensor network linear layer.

    Weight is factorized as W ~ (A kron B) @ C with a low-rank root, giving a
    binary-tree structure: two leaf cores feed a root core.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bond_dimension: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        chi = bond_dimension
        self.left = nn.Parameter(torch.randn(out_features, chi) / math.sqrt(in_features))
        self.root = nn.Parameter(torch.eye(chi))
        self.right = nn.Parameter(torch.randn(chi, in_features) / math.sqrt(chi))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, bond_dimension: int = 32) -> "TTNSLinear":
        t = cls(linear.in_features, linear.out_features, bond_dimension, linear.bias is not None)
        U, S, Vh = torch.linalg.svd(linear.weight.data, full_matrices=False)
        chi = min(bond_dimension, S.shape[0])
        t.left.data = U[:, :chi].contiguous()
        t.root.data = torch.diag(S[:chi]).contiguous()
        t.right.data = Vh[:chi].contiguous()
        if linear.bias is not None:
            assert t.bias is not None
            t.bias.data.copy_(linear.bias.data)
        return t

    def reconstruct_weight(self) -> torch.Tensor:
        return self.left @ self.root @ self.right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # contract right-to-left: avoids materializing full W
        h = nn.functional.linear(x, self.right)
        h = h @ self.root.T
        out = h @ self.left.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def get_compression_ratio(self) -> float:
        orig = self.in_features * self.out_features + (
            self.out_features if self.bias is not None else 0
        )
        return orig / sum(p.numel() for p in self.parameters())


TTNSCompressor = TTNSLinear


@dataclass
class ModelCompressionReport:
    layer_stats: Dict[str, CompressionStats] = field(default_factory=dict)

    @property
    def total_original(self) -> int:
        return sum(s.original_params for s in self.layer_stats.values())

    @property
    def total_compressed(self) -> int:
        return sum(s.compressed_params for s in self.layer_stats.values())

    @property
    def overall_ratio(self) -> float:
        return self.total_original / max(self.total_compressed, 1)


def compress_model(
    model: nn.Module,
    bond_dimension: int = 32,
    method: str = "mps",
    min_features: int = 64,
    skip_names: Sequence[str] = ("lm_head", "embed"),
) -> Tuple[nn.Module, ModelCompressionReport]:
    """Replace eligible nn.Linear layers in-place with tensor-network layers.

    Layers smaller than `min_features` in either dimension, or whose qualified
    name contains any of `skip_names`, are left untouched (embedding/output
    heads are accuracy-critical, mirroring CompactifAI practice).
    """
    report = ModelCompressionReport()
    replacements: List[Tuple[nn.Module, str, nn.Module, str]] = []
    for qual_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full = f"{qual_name}.{child_name}" if qual_name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if any(s in full for s in skip_names):
                continue
            if min(child.in_features, child.out_features) < min_features:
                continue
            if method == "mps":
                new = MPSLinear.from_linear(child, bond_dimension=bond_dimension)
                stats = new.stats
            else:
                new = TTNSLinear.from_linear(child, bond_dimension=bond_dimension)
                W = child.weight.data
                rel = (torch.norm(W - new.reconstruct_weight()) / torch.norm(W)).item()
                orig = child.weight.numel() + (child.bias.numel() if child.bias is not None else 0)
                stats = CompressionStats(orig, sum(p.numel() for p in new.parameters()),
                                         bond_dimension, rel)
            if stats is not None and stats.compression_ratio > 1.0:
                replacements.append((module, child_name, new, full))
                report.layer_stats[full] = stats
    for parent, name, new, _ in replacements:
        setattr(parent, name, new)
    logger.info("Compressed %d layers, overall ratio %.2fx",
                len(replacements), report.overall_ratio if report.layer_stats else 1.0)
    return model, report
