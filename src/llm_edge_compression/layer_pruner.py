from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class LayerPruningResult:
    total_layers: int
    removed_layers: list[int]
    strategy: str
    angular_distance: float | None = None


def get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Return the Transformer block list for common decoder-only HF models."""
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
        ("blocks", None),
    )
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        if layers_name is None and isinstance(parent, nn.ModuleList):
            return parent
        if parent is not None:
            layers = getattr(parent, layers_name, None)
            if isinstance(layers, nn.ModuleList):
                return layers

    direct = getattr(model, "layers", None)
    if isinstance(direct, nn.ModuleList):
        return direct

    raise ValueError(
        "Could not locate Transformer blocks. Supported layouts include "
        "model.layers, transformer.h, transformer.layers, gpt_neox.layers, and blocks."
    )


def _replace_transformer_layers(model: nn.Module, new_layers: nn.ModuleList) -> None:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
        ("blocks", None),
    )
    for parent_name, layers_name in candidates:
        parent = getattr(model, parent_name, None)
        if layers_name is None:
            if parent is not None and parent is get_transformer_layers(model):
                setattr(model, parent_name, new_layers)
                return
        elif parent is not None and getattr(parent, layers_name, None) is get_transformer_layers(model):
            setattr(parent, layers_name, new_layers)
            _update_layer_count(parent, len(new_layers))
            return

    if getattr(model, "layers", None) is get_transformer_layers(model):
        model.layers = new_layers
        _update_layer_count(model, len(new_layers))
        return

    raise RuntimeError("Unable to replace Transformer block list.")


def _update_layer_count(container: nn.Module, count: int) -> None:
    config = getattr(container, "config", None)
    if config is not None and hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = count


def _last_token_representation(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 2:
        return hidden[:, -1, :] if hidden.shape[1] != 1 else hidden
    if hidden.ndim == 3:
        return hidden[:, -1, :]
    raise ValueError(f"Unexpected hidden-state shape: {tuple(hidden.shape)}")


@torch.no_grad()
def collect_layer_representations(
    model: nn.Module,
    calibration_batches: Iterable[dict[str, torch.Tensor]],
) -> list[torch.Tensor]:
    """Collect mean final-token inputs to every Transformer block."""
    layers = get_transformer_layers(model)
    captured: list[list[torch.Tensor]] = [[] for _ in layers]
    hooks = []

    def make_hook(index: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
            if not inputs:
                return
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                return
            rep = _last_token_representation(hidden).detach().float().mean(dim=0).cpu()
            captured[index].append(rep)
        return hook

    for index, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(index)))

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        for batch in calibration_batches:
            batch_on_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            model(**batch_on_device)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    if not all(captured):
        raise RuntimeError("No layer representations were captured during calibration.")

    return [torch.stack(values).mean(dim=0) for values in captured]


def angular_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Angular distance in radians between two hidden representations."""
    x = x.float().reshape(1, -1)
    y = y.float().reshape(1, -1)
    cosine = F.cosine_similarity(x, y, dim=-1).clamp(-1.0, 1.0)
    return float(torch.acos(cosine).item())


def select_layers_to_prune(
    representations: Sequence[torch.Tensor],
    num_remove: int,
    strategy: str = "similarity",
    seed: int = 0,
) -> tuple[list[int], float | None]:
    """Select a contiguous block while protecting the final Transformer block."""
    total_layers = len(representations)
    if num_remove < 1:
        raise ValueError("num_remove must be at least 1.")
    if num_remove >= total_layers:
        raise ValueError("Cannot remove all Transformer blocks.")
    if strategy not in {"similarity", "deepest", "random"}:
        raise ValueError("strategy must be one of: similarity, deepest, random")

    max_start = total_layers - num_remove - 1

    if strategy == "deepest":
        start = max_start
        return list(range(start, start + num_remove)), None

    if strategy == "random":
        generator = torch.Generator().manual_seed(seed)
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        return list(range(start, start + num_remove)), None

    best_start = 0
    best_distance = float("inf")
    for start in range(max_start + 1):
        distance = angular_distance(
            representations[start],
            representations[start + num_remove],
        )
        if distance < best_distance:
            best_distance = distance
            best_start = start

    return list(range(best_start, best_start + num_remove)), best_distance


class LayerPruningCompressor:
    """Similarity-guided or deepest-first structured Transformer layer pruning."""

    def __init__(
        self,
        num_remove: int | None = None,
        prune_ratio: float | None = None,
        strategy: str = "similarity",
        seed: int = 0,
        selected_layers: Sequence[int] | None = None,
    ) -> None:
        if num_remove is None and prune_ratio is None and selected_layers is None:
            raise ValueError("Provide num_remove, prune_ratio, or selected_layers.")
        if prune_ratio is not None and not 0.0 < prune_ratio < 1.0:
            raise ValueError("prune_ratio must be between 0 and 1.")
        self.num_remove = num_remove
        self.prune_ratio = prune_ratio
        self.strategy = strategy
        self.seed = seed
        self.selected_layers = list(selected_layers) if selected_layers is not None else None
        self.result: LayerPruningResult | None = None

    def fit(self, model: nn.Module, calibration_batches: Iterable[dict[str, torch.Tensor]]) -> LayerPruningResult:
        total_layers = len(get_transformer_layers(model))
        num_remove = self.num_remove
        if num_remove is None:
            num_remove = max(1, int(total_layers * float(self.prune_ratio)))
        if num_remove >= total_layers:
            raise ValueError("Pruning configuration would remove all Transformer blocks.")

        if self.selected_layers is not None:
            removed = list(self.selected_layers)
            distance = None
        elif self.strategy == "deepest":
            removed, distance = select_layers_to_prune(
                [torch.empty(1)] * total_layers, num_remove, strategy="deepest"
            )
        else:
            representations = collect_layer_representations(model, calibration_batches)
            removed, distance = select_layers_to_prune(
                representations,
                num_remove,
                strategy=self.strategy,
                seed=self.seed,
            )

        if any(i < 0 or i >= total_layers - 1 for i in removed):
            raise ValueError("Selected layers must be valid and must not include the final Transformer block.")
        self.selected_layers = removed
        self.result = LayerPruningResult(total_layers, removed, self.strategy, distance)
        return self.result

    def compress(self, model: nn.Module) -> nn.Module:
        layers = get_transformer_layers(model)
        if self.selected_layers is None:
            raise RuntimeError("Call fit() before compress() when layers are not supplied explicitly.")
        selected = set(self.selected_layers)
        if any(i < 0 or i >= len(layers) - 1 for i in selected):
            raise ValueError("Selected layers must be valid and must not include the final Transformer block.")
        remaining = [layer for i, layer in enumerate(layers) if i not in selected]
        _replace_transformer_layers(model, nn.ModuleList(remaining))
        return model
