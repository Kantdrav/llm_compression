from __future__ import annotations

import torch
from torch import nn
import pytest

from llm_edge_compression.adaptive_mpo import (
    AdaptiveLayerCompressionReport,
    AdaptiveMPOCompressor,
    AdaptiveMPOLinear,
    compare_mpo_compressors,
)
from llm_edge_compression.config import CompressionPolicy

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None


class SimpleTestModel(nn.Module):
    def __init__(self, in_features: int = 64, out_features: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_adaptive_mpo_linear_2_site():
    torch.manual_seed(42)
    layer = nn.Linear(64, 48)
    mpo_layer = AdaptiveMPOLinear.from_linear(
        layer,
        energy_threshold=0.99,
        mpo_sites=2,
    )

    assert len(mpo_layer.cores) == 2
    assert len(mpo_layer.selected_ranks) == 1
    assert mpo_layer.selected_ranks[0] >= 1

    x = torch.randn(2, 5, 64)
    out_mpo = mpo_layer(x)
    assert out_mpo.shape == (2, 5, 48)

    # Direct core reconstruction check
    w_recon = mpo_layer.reconstruct_weight()
    assert w_recon.shape == (48, 64)
    assert mpo_layer.relative_frobenius_error >= 0.0


def test_adaptive_mpo_linear_3_site():
    torch.manual_seed(42)
    layer = nn.Linear(64, 64)
    mpo_layer = AdaptiveMPOLinear.from_linear(
        layer,
        energy_threshold=0.99,
        mpo_sites=3,
    )

    assert len(mpo_layer.cores) == 3
    assert len(mpo_layer.selected_ranks) == 2
    assert all(r >= 1 for r in mpo_layer.selected_ranks)

    x = torch.randn(3, 64)
    out = mpo_layer(x)
    assert out.shape == (3, 64)


def test_adaptive_mpo_linear_4_site():
    torch.manual_seed(42)
    layer = nn.Linear(16, 16)
    mpo_layer = AdaptiveMPOLinear.from_linear(
        layer,
        energy_threshold=0.95,
        mpo_sites=4,
    )

    assert len(mpo_layer.cores) == 4
    assert len(mpo_layer.selected_ranks) == 3

    x = torch.randn(2, 16)
    out = mpo_layer(x)
    assert out.shape == (2, 16)


def test_adaptive_mpo_monotonic_energy_threshold():
    """Verify that increasing the energy threshold never decreases the selected rank."""
    torch.manual_seed(42)
    weight = torch.randn(64, 64)

    thresholds = [0.80, 0.90, 0.98, 0.999]
    ranks_per_cut: list[list[int]] = []
    errors: list[float] = []

    for eta in thresholds:
        mpo = AdaptiveMPOLinear._from_weight_adaptive(
            weight,
            None,
            energy_threshold=eta,
            mpo_sites=3,
        )
        ranks_per_cut.append(list(mpo.selected_ranks))
        errors.append(mpo.relative_frobenius_error)

    # Check monotonicity across each cut
    num_cuts = len(ranks_per_cut[0])
    for cut_idx in range(num_cuts):
        for i in range(len(thresholds) - 1):
            assert ranks_per_cut[i][cut_idx] <= ranks_per_cut[i + 1][cut_idx], (
                f"Cut {cut_idx}: rank decreased from {ranks_per_cut[i][cut_idx]} "
                f"to {ranks_per_cut[i+1][cut_idx]} as energy increased from {thresholds[i]} to {thresholds[i+1]}"
            )

    # Higher energy threshold must result in lower or equal reconstruction error
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-6


def test_adaptive_mpo_max_bond_dim_limits_rank_and_flags_unmet():
    """Verify that max_bond_dim clamps rank and correctly flags when threshold is unmet."""
    torch.manual_seed(42)
    weight = torch.randn(64, 64)

    # High energy threshold with strict max_bond_dim = 2
    mpo = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=0.9999,
        max_bond_dim=2,
        mpo_sites=3,
    )

    assert all(r <= 2 for r in mpo.selected_ranks)
    assert mpo.energy_threshold_met is False


def test_adaptive_mpo_min_bond_dim():
    """Verify that min_bond_dim enforces lower bound even on low-rank inputs."""
    torch.manual_seed(42)
    # Rank-1 outer product matrix
    a = torch.randn(64, 1)
    b = torch.randn(64, 1)
    weight = a @ b.t()

    mpo = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=0.99,
        min_bond_dim=4,
        mpo_sites=3,
    )

    assert all(r >= 4 for r in mpo.selected_ranks)


def test_adaptive_mpo_zero_weights():
    """Verify numerical stability on zero matrices."""
    weight = torch.zeros(64, 64)
    mpo = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=0.99,
        mpo_sites=3,
    )

    assert all(r == 1 for r in mpo.selected_ranks)
    assert mpo.relative_frobenius_error == 0.0
    x = torch.randn(2, 64)
    out = mpo(x)
    assert torch.allclose(out, torch.zeros(2, 64))


def test_adaptive_mpo_near_zero_weights():
    """Verify numerical stability on sub-epsilon weights."""
    weight = torch.randn(64, 64) * 1e-16
    mpo = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=0.99,
        mpo_sites=3,
    )

    assert all(r == 1 for r in mpo.selected_ranks)
    assert not torch.isnan(mpo.cores[0]).any()


def test_adaptive_mpo_low_rank_matrix_exactness():
    """Verify high reconstruction accuracy on intrinsically low-rank weights."""
    torch.manual_seed(42)
    a = torch.randn(64, 2)
    b = torch.randn(64, 2)
    weight = a @ b.t()

    mpo = AdaptiveMPOLinear._from_weight_adaptive(
        weight,
        None,
        energy_threshold=0.9999,
        mpo_sites=3,
    )

    assert mpo.relative_frobenius_error < 1e-4
    assert mpo.energy_threshold_met is True


@pytest.mark.skipif(Conv1D is None, reason="transformers Conv1D not installed")
def test_adaptive_mpo_conv1d_compatibility():
    """Verify compatibility with GPT-2 Conv1D projection layers."""
    torch.manual_seed(42)
    conv = Conv1D(64, 32)
    mpo = AdaptiveMPOLinear.from_conv1d(
        conv,
        energy_threshold=0.99,
        mpo_sites=3,
    )

    x = torch.randn(2, 32)
    out = mpo(x)
    assert out.shape == (2, 64)


def test_adaptive_mpo_compressor_full_model_and_reporting():
    """Verify AdaptiveMPOCompressor on a multi-layer model with parameter reporting."""
    torch.manual_seed(42)
    model = SimpleTestModel(in_features=64, out_features=64).eval()
    orig_params = sum(p.numel() for p in model.parameters())

    compressor = AdaptiveMPOCompressor(
        energy_threshold=0.95,
        target_reduction=0.20,
        mpo_sites=3,
        min_bond_dim=1,
    )
    compressed = compressor.compress(model)

    assert isinstance(compressed.fc1, AdaptiveMPOLinear)
    assert isinstance(compressed.fc2, AdaptiveMPOLinear)

    # Check compression reports
    assert "fc1" in compressor.compression_reports
    assert "fc2" in compressor.compression_reports

    report_fc1: AdaptiveLayerCompressionReport = compressor.compression_reports["fc1"]
    assert report_fc1.module_path == "fc1"
    assert report_fc1.original_parameters == 64 * 64 + 64
    assert report_fc1.compressed_parameters < report_fc1.original_parameters
    assert len(report_fc1.selected_ranks) == 2
    assert report_fc1.target_reduction == 0.20
    assert isinstance(report_fc1.target_reduction_achieved, bool)

    # Test forward pass
    x = torch.randn(4, 64)
    out = compressed(x)
    assert out.shape == (4, 32)


def test_adaptive_mpo_compressor_layer_policy_skip():
    """Verify that layer policy skip patterns are respected."""
    model = SimpleTestModel().eval()
    policy = CompressionPolicy(skip_module_patterns=(r"^fc1$",))

    compressor = AdaptiveMPOCompressor(layer_policy=policy, energy_threshold=0.95)
    compressed = compressor.compress(model)

    assert isinstance(compressed.fc1, nn.Linear)
    assert isinstance(compressed.fc2, AdaptiveMPOLinear)
    assert "fc1" not in compressor.compression_reports
    assert "fc2" in compressor.compression_reports


def test_compare_mpo_compressors_utility():
    """Verify the comparison utility between fixed and adaptive MPO."""
    torch.manual_seed(42)
    weight = torch.randn(64, 48)

    comparison = compare_mpo_compressors(
        weight,
        mpo_sites=3,
        energy_threshold=0.99,
        rank_ratio=0.5,
    )

    assert comparison["original_shape"] == (64, 48)
    assert comparison["original_parameters"] == 64 * 48
    assert "fixed_mpo" in comparison
    assert "adaptive_mpo" in comparison

    fixed = comparison["fixed_mpo"]
    adaptive = comparison["adaptive_mpo"]

    assert fixed["compressed_parameters"] > 0
    assert adaptive["compressed_parameters"] > 0
    assert len(adaptive["selected_ranks"]) == 2
    assert len(adaptive["retained_energies"]) == 2
    assert adaptive["relative_frobenius_error"] >= 0.0


def test_independent_forward_equivalence_linear():
    """Verify that MPO forward pass strictly equals dense F.linear with reconstructed weights."""
    import torch.nn.functional as F
    torch.manual_seed(101)

    layer = nn.Linear(64, 48, bias=True)
    mpo = AdaptiveMPOLinear.from_linear(layer, energy_threshold=0.95, mpo_sites=3)

    # Independent basis probe to extract effective dense weight
    eye = torch.eye(64)
    with torch.no_grad():
        w_probe = (mpo(eye) - mpo.bias).t()

    # Compare forward pass on batched 3D tensor
    x = torch.randn(4, 7, 64)
    y_mpo = mpo(x)
    y_dense = F.linear(x, w_probe, mpo.bias)

    max_abs_diff = torch.max(torch.abs(y_mpo - y_dense)).item()
    assert max_abs_diff < 1e-5, f"Discrepancy between MPO forward and F.linear: {max_abs_diff}"


@pytest.mark.skipif(Conv1D is None, reason="transformers Conv1D not installed")
def test_independent_forward_equivalence_conv1d():
    """Verify that Conv1D MPO forward pass strictly equals dense F.linear."""
    import torch.nn.functional as F
    torch.manual_seed(102)

    conv = Conv1D(48, 64)  # out=48, in=64
    mpo = AdaptiveMPOLinear.from_conv1d(conv, energy_threshold=0.95, mpo_sites=3)

    eye = torch.eye(64)
    with torch.no_grad():
        w_probe = (mpo(eye) - mpo.bias).t()

    x = torch.randn(3, 5, 64)
    y_mpo = mpo(x)
    y_dense = F.linear(x, w_probe, mpo.bias)

    max_abs_diff = torch.max(torch.abs(y_mpo - y_dense)).item()
    assert max_abs_diff < 1e-5, f"Discrepancy for Conv1D: {max_abs_diff}"


def test_parameter_accounting_direct_from_shapes():
    """Verify parameter counts calculated directly from core shapes match parameter tensors."""
    torch.manual_seed(103)
    layer = nn.Linear(72, 48, bias=True)
    mpo = AdaptiveMPOLinear.from_linear(layer, energy_threshold=0.95, mpo_sites=3)

    # Direct calculation from shapes:
    shape_param_count = sum(core.numel() for core in mpo.cores) + mpo.bias.numel()
    tensor_param_count = sum(p.numel() for p in mpo.parameters())
    dense_param_count = layer.weight.numel() + layer.bias.numel()

    assert shape_param_count == tensor_param_count
    assert shape_param_count > 0
    assert dense_param_count == 48 * 72 + 48


def test_multi_site_independent_reconstruction():
    """Verify independent reconstruction across 2, 3, and 4 sites for full-rank and low-rank matrices."""
    torch.manual_seed(104)

    for sites in [2, 3, 4]:
        dim = 16 if sites == 4 else 64

        # 1. Full-rank random matrix
        w_full = torch.randn(dim, dim)
        mpo_full = AdaptiveMPOLinear._from_weight_adaptive(w_full, None, energy_threshold=0.99, mpo_sites=sites)
        w_recon_full = mpo_full.reconstruct_weight()
        err_full = (torch.norm(w_full - w_recon_full) / torch.norm(w_full)).item()
        assert abs(err_full - mpo_full.relative_frobenius_error) < 1e-6
        assert len(mpo_full.selected_ranks) == sites - 1

        # 2. Deliberately low-TT-rank matrix (sum of 2 Kronecker products across the cut)
        dim_part = int(dim ** 0.5) if sites == 2 else (2 if sites == 4 else 8)
        rest_part = dim // dim_part
        a1, b1 = torch.randn(dim_part, dim_part), torch.randn(rest_part, rest_part)
        a2, b2 = torch.randn(dim_part, dim_part), torch.randn(rest_part, rest_part)
        w_low_tt = torch.kron(a1, b1) + torch.kron(a2, b2)

        mpo_low = AdaptiveMPOLinear._from_weight_adaptive(w_low_tt, None, energy_threshold=0.9999, mpo_sites=sites)
        w_recon_low = mpo_low.reconstruct_weight()
        err_low = (torch.norm(w_low_tt - w_recon_low) / torch.norm(w_low_tt)).item()
        assert err_low < 1e-4, f"Low-TT-rank reconstruction error too high for {sites} sites: {err_low}"


def test_independent_retained_energy_audit():
    """Independently calculate SVD on unfolding matricization and verify recorded retained energy."""
    from llm_edge_compression.compressors import _partition_factors
    torch.manual_seed(105)

    w = torch.randn(64, 64)
    mpo = AdaptiveMPOLinear._from_weight_adaptive(w, None, energy_threshold=0.95, mpo_sites=3)

    out_factors = _partition_factors(64, 3)
    in_factors = _partition_factors(64, 3)
    tensor = w.reshape(*out_factors, *in_factors)
    perm = [0, 3, 1, 4, 2, 5]
    remainder = tensor.permute(*perm).contiguous()

    # Cut 0 audit
    m0 = remainder.reshape(out_factors[0] * in_factors[0], -1)
    s0 = torch.linalg.svdvals(m0)
    total_e0 = s0.square().sum()
    r0 = mpo.selected_ranks[0]
    expected_e0 = float((s0[:r0].square().sum() / total_e0).item())

    assert abs(expected_e0 - mpo.retained_energies[0]) < 1e-5


def test_left_canonical_orthogonality():
    """Verify that every non-terminal core is strictly left-orthogonal (U^T U = I)."""
    torch.manual_seed(201)

    for sites in [2, 3, 4]:
        dim = 16 if sites == 4 else 64
        w = torch.randn(dim, dim)
        mpo = AdaptiveMPOLinear._from_weight_adaptive(
            w,
            None,
            energy_threshold=0.95,
            mpo_sites=sites,
        )

        # For every non-terminal core k in 0 .. sites-2:
        for k in range(sites - 1):
            core = mpo.cores[k]
            # Core shape is (left_rank, out_dim, in_dim, right_rank)
            r_left, o_k, i_k, r_right = core.shape
            unfolding = core.reshape(r_left * o_k * i_k, r_right)
            gram = unfolding.t() @ unfolding
            eye = torch.eye(r_right)

            max_ortho_dev = torch.max(torch.abs(gram - eye)).item()
            assert max_ortho_dev < 1e-5, f"Core {k} (sites={sites}) violated left-orthogonality with deviation {max_ortho_dev}"


def test_tt_svd_global_error_bound():
    """Verify Oseledets global error bound: actual_error^2 <= sum(epsilon_k^2)."""
    torch.manual_seed(202)

    for sites in [2, 3, 4]:
        dim = 16 if sites == 4 else 64

        # 1. Truncated case (eta = 0.90)
        w_trunc = torch.randn(dim, dim)
        mpo_trunc = AdaptiveMPOLinear._from_weight_adaptive(w_trunc, None, energy_threshold=0.90, mpo_sites=sites)

        act_err_sq = mpo_trunc.actual_reconstruction_error_squared
        bound_sq = mpo_trunc.theoretical_error_bound_squared
        assert bound_sq >= 0.0
        # Check actual error <= theoretical bound + numerical slack
        assert act_err_sq <= bound_sq * (1.0 + 1e-4) + 1e-6, (
            f"Oseledets bound violated for sites={sites}: actual={act_err_sq}, bound={bound_sq}"
        )

        # 2. Untruncated full-rank case (eta = 1.0)
        mpo_full = AdaptiveMPOLinear._from_weight_adaptive(w_trunc, None, energy_threshold=1.0, mpo_sites=sites)
        assert mpo_full.relative_frobenius_error < 1e-4
        assert sum(mpo_full.discarded_energies) < 1e-4
