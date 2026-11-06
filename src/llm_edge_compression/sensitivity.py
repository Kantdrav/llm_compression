from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn

from .config import CompressionPolicy
from .paper_mpo import ResearchMPOLinear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:  # pragma: no cover - optional dependency shape varies by transformers version
    Conv1D = None


@dataclass(slots=True)
class LayerSensitivityResult:
    module_path: str
    bond_dim: int
    original_parameters: int
    compressed_parameters: int
    parameter_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int
    size_ratio: float
    output_mse: float
    normalized_output_mse: float


def profile_mpo_sensitivity(
    model: nn.Module,
    calibration_batches: Iterable[object],
    *,
    bond_dims: Sequence[int] = (4, 8, 16),
    mpo_sites: int = 3,
    layer_policy: CompressionPolicy | None = None,
) -> list[LayerSensitivityResult]:
    """Measure one-layer MPO distortion at several bond dimensions.

    Each candidate is evaluated from an untouched copy of ``model``. Only one
    eligible linear layer is replaced, so the resulting output error measures
    that layer's local sensitivity rather than cumulative compression error.
    Calibration batches are materialized once because every candidate needs the
    same observations.
    """

    if not bond_dims or any(int(value) < 1 for value in bond_dims):
        raise ValueError("bond_dims must contain at least one positive integer")
    if mpo_sites < 2:
        raise ValueError("mpo_sites must be at least 2")

    batches = list(calibration_batches)
    if not batches:
        raise ValueError("calibration_batches must contain at least one batch")

    policy = layer_policy or CompressionPolicy()
    candidates = _candidate_paths(model, policy)
    if not candidates:
        return []

    reference = copy.deepcopy(model).eval()
    device = _model_device(reference)
    reference = reference.to(device)
    with torch.no_grad():
        reference_outputs = [_extract_tensor(_forward_model(reference, _move_batch(batch, device))).detach() for batch in batches]

    original_parameters = _count_parameters(model)
    original_size = _state_dict_size(model)
    results: list[LayerSensitivityResult] = []
    for module_path, module_type in candidates:
        for bond_dim in bond_dims:
            candidate = copy.deepcopy(model).eval().to(device)
            parent, child_name = _resolve_parent(candidate, module_path)
            child = getattr(parent, child_name)
            if module_type == "linear":
                compressed = ResearchMPOLinear.from_linear(
                    child,
                    bond_dim=int(bond_dim),
                    mpo_sites=mpo_sites,
                )
            else:
                compressed = ResearchMPOLinear.from_conv1d(
                    child,
                    bond_dim=int(bond_dim),
                    mpo_sites=mpo_sites,
                )
            setattr(parent, child_name, compressed)

            with torch.no_grad():
                errors = []
                for batch, expected in zip(batches, reference_outputs):
                    actual = _extract_tensor(_forward_model(candidate, _move_batch(batch, device)))
                    errors.append(torch.mean((actual - expected) ** 2).item())
                output_mse = sum(errors) / len(errors)
                reference_scale = sum(torch.mean(expected.square()).item() for expected in reference_outputs)
                reference_scale /= len(reference_outputs)

            compressed_parameters = _count_parameters(candidate)
            compressed_size = _state_dict_size(candidate)
            results.append(
                LayerSensitivityResult(
                    module_path=module_path,
                    bond_dim=int(bond_dim),
                    original_parameters=original_parameters,
                    compressed_parameters=compressed_parameters,
                    parameter_ratio=compressed_parameters / max(original_parameters, 1),
                    original_size_bytes=original_size,
                    compressed_size_bytes=compressed_size,
                    size_ratio=compressed_size / max(original_size, 1),
                    output_mse=output_mse,
                    normalized_output_mse=output_mse / max(reference_scale, torch.finfo(torch.float32).eps),
                )
            )
    return results


def write_sensitivity_profile(results: Iterable[LayerSensitivityResult], path: Path) -> Path:
    """Write a JSON sensitivity profile suitable for later rank allocation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _candidate_paths(
    model: nn.Module,
    policy: CompressionPolicy,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for module_path, module in model.named_modules():
        if not module_path or _is_skipped(module_path, policy):
            continue
        if isinstance(module, nn.Linear):
            candidates.append((module_path, "linear"))
        elif Conv1D is not None and isinstance(module, Conv1D):
            candidates.append((module_path, "conv1d"))
    return candidates


def _is_skipped(module_path: str, policy: CompressionPolicy) -> bool:
    return any(re.search(pattern, module_path) for pattern in policy.skip_module_patterns)


def _resolve_parent(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_dict_size(model: nn.Module) -> int:
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def _forward_model(model: nn.Module, inputs: object) -> object:
    if isinstance(inputs, torch.Tensor):
        return model(inputs)
    if isinstance(inputs, dict):
        try:
            return model(**inputs)
        except TypeError:
            for value in inputs.values():
                if isinstance(value, torch.Tensor):
                    return model(value)
            return model(inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _move_batch(batch: object, device: torch.device) -> object:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in batch)
    if isinstance(batch, list):
        return [value.to(device) if isinstance(value, torch.Tensor) else value for value in batch]
    return batch


def _extract_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict):
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(output)!r}")
