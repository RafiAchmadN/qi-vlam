import copy

import pytest
import torch
import torch.nn as nn

from src.compression.tensor_network import (
    MPSLinear, TTNSLinear, _factorize, compress_model,
)
from src.compression.quantization import Int8Linear, quantize_model_int8


class TestFactorize:
    def test_product(self):
        for n in (256, 128, 96, 60):
            f = _factorize(n, 3)
            assert len(f) == 3
            assert torch.prod(torch.tensor(f)).item() == n

    def test_llava_mlp_dims_no_degenerate_factor(self):
        # 4096 and 11008 are the LLaVA-1.5-7B MLP in/out dims (CLAUDE.md
        # invariant #3): a factor of 1 would silently degrade MPS to fewer
        # effective cores, so pin exact-product + no-1 across factor counts.
        for n in (4096, 11008):
            for num_factors in (2, 3, 4):
                f = _factorize(n, num_factors)
                assert len(f) == num_factors
                assert torch.prod(torch.tensor(f)).item() == n
                assert 1 not in f


class TestMPSLinear:
    def test_forward_shape(self):
        m = MPSLinear(256, 128, bond_dimension=16)
        y = m(torch.randn(4, 256))
        assert y.shape == (4, 128)

    def test_from_linear_reconstruction(self):
        lin = nn.Linear(64, 64)
        # full-rank bond dim -> near-exact reconstruction
        m = MPSLinear.from_linear(lin, bond_dimension=64, num_cores=2)
        W_hat = m.reconstruct_weight()
        assert torch.allclose(W_hat, lin.weight.data, atol=1e-4)

    def test_from_linear_output_agreement(self):
        lin = nn.Linear(64, 64)
        m = MPSLinear.from_linear(lin, bond_dimension=64, num_cores=2)
        x = torch.randn(8, 64)
        assert torch.allclose(m(x), lin(x), atol=1e-3)

    def test_compression_ratio_gt_one(self):
        lin = nn.Linear(256, 256)
        m = MPSLinear.from_linear(lin, bond_dimension=8)
        assert m.get_compression_ratio() > 2.0

    def test_gradients_flow(self):
        m = MPSLinear(64, 32, bond_dimension=8)
        m(torch.randn(2, 64)).sum().backward()
        assert all(p.grad is not None for p in m.cores)

    def test_stats_populated(self):
        m = MPSLinear.from_linear(nn.Linear(128, 128), bond_dimension=16)
        assert m.stats is not None
        assert 0 <= m.stats.relative_error <= 1.5


class TestTTNSLinear:
    def test_forward_and_svd_equivalence(self):
        lin = nn.Linear(96, 96)
        t = TTNSLinear.from_linear(lin, bond_dimension=96)
        x = torch.randn(5, 96)
        assert torch.allclose(t(x), lin(x), atol=1e-4)

    def test_low_rank_compression(self):
        t = TTNSLinear.from_linear(nn.Linear(256, 256), bond_dimension=16)
        assert t.get_compression_ratio() > 4.0


class TestCompressModel:
    def _model(self):
        return nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
                             nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8))

    def test_replaces_large_layers_only(self):
        m, report = compress_model(self._model(), bond_dimension=8)
        assert len(report.layer_stats) == 2  # final 128->8 skipped (min_features)
        assert report.overall_ratio > 1.5

    def test_forward_still_works(self):
        m, _ = compress_model(self._model(), bond_dimension=16)
        assert m(torch.randn(3, 128)).shape == (3, 8)

    def test_ttns_method(self):
        m, report = compress_model(self._model(), bond_dimension=8, method="ttns")
        assert report.overall_ratio > 1.5


class TestDeterminism:
    def test_same_seed_same_output(self):
        torch.manual_seed(42)
        a = MPSLinear(64, 64, bond_dimension=8)
        torch.manual_seed(42)
        b = MPSLinear(64, 64, bond_dimension=8)
        x = torch.randn(2, 64)
        assert torch.equal(a(x), b(x))


class TestInt8Baseline:
    def test_quantized_close_to_fp32(self):
        lin = nn.Linear(128, 64)
        q = Int8Linear(lin)
        x = torch.randn(4, 128)
        assert torch.allclose(q(x), lin(x), atol=0.1)

    def test_quantize_model(self):
        m = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 8))
        quantize_model_int8(m)
        assert isinstance(m[0], Int8Linear)
        assert isinstance(m[1], nn.Linear)  # below min_features

    def test_storage_bytes(self):
        q = Int8Linear(nn.Linear(128, 128))
        assert q.effective_param_bytes() < 128 * 128 * 4


class TestDirectContraction:
    def test_matches_reconstructed_weight(self):
        lin = nn.Linear(96, 60)
        m = MPSLinear.from_linear(lin, bond_dimension=999, num_cores=3)
        x = torch.randn(7, 96)
        y_direct = m(x)
        y_dense = nn.functional.linear(x, m.reconstruct_weight(), m.bias)
        assert torch.allclose(y_direct, y_dense, atol=1e-4)

    def test_3d_input(self):
        m = MPSLinear.from_linear(nn.Linear(64, 32), bond_dimension=8)
        assert m(torch.randn(2, 5, 64)).shape == (2, 5, 32)
