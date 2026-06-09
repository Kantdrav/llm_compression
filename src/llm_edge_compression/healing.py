from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class HealingConfig:
    steps: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.0


def heal_model(
    student_model: nn.Module,
    teacher_model: nn.Module,
    calibration_batches: Iterable[object],
    config: HealingConfig | None = None,
) -> nn.Module:
    config = config or HealingConfig()

    device = next(student_model.parameters()).device
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device).eval()
    student_model.train()

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for step, batch in enumerate(calibration_batches):
        if step >= config.steps:
            break

        inputs = _move_batch_to_device(batch, device)

        with torch.no_grad():
            teacher_outputs = _extract_tensor(_forward_model(teacher_model, inputs))

        student_outputs = _extract_tensor(_forward_model(student_model, inputs))
        loss = F.mse_loss(student_outputs, teacher_outputs)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    student_model.eval()
    return student_model


def _forward_model(model: nn.Module, inputs: object) -> object:
    if isinstance(inputs, torch.Tensor):
        return model(inputs)
    if isinstance(inputs, dict):
        try:
            return model(**inputs)
        except TypeError:
            # Fallback: use the first tensor-like value in the dict as positional input
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    return model(v)
            # Last resort: pass the dict itself
            return model(inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _move_batch_to_device(batch: object, device: torch.device) -> object:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in batch)
    if isinstance(batch, list):
        return [item.to(device) if isinstance(item, torch.Tensor) else item for item in batch]
    return batch


def _extract_tensor(output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor] | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        logits = getattr(output, "logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(output, (tuple, list)) and output:
        first_item = output[0]
        if isinstance(first_item, torch.Tensor):
            return first_item
    if isinstance(output, dict):
        logits = output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
        for value in output.values():
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(f"Unsupported model output type: {type(output)!r}")