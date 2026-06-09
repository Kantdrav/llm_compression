from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from llm_edge_compression.compressors import TensorNetworkCompressor
from llm_edge_compression.healing import HealingConfig, heal_model


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def test_healing_reduces_teacher_student_gap() -> None:
    torch.manual_seed(0)

    teacher = TinyModel().eval()
    student = TinyModel().eval()
    student.load_state_dict(teacher.state_dict())

    compressed = TensorNetworkCompressor(rank_ratio=0.4).compress(student)
    batches = [torch.randn(4, 16) for _ in range(8)]

    with torch.no_grad():
        before = sum(F.mse_loss(compressed(batch), teacher(batch)).item() for batch in batches) / len(batches)

    healed = heal_model(
        compressed,
        teacher,
        batches,
        HealingConfig(steps=40, learning_rate=5e-3),
    )

    with torch.no_grad():
        after = sum(F.mse_loss(healed(batch), teacher(batch)).item() for batch in batches) / len(batches)

    assert after < before


def test_healing_accepts_batch_dictionaries() -> None:
    torch.manual_seed(1)

    teacher = TinyModel().eval()
    student = TinyModel().eval()
    student.load_state_dict(teacher.state_dict())
    compressed = TensorNetworkCompressor(rank_ratio=0.4).compress(student)

    batches = [{"input_ids": torch.randn(4, 16)} for _ in range(4)]

    healed = heal_model(compressed, teacher, batches, HealingConfig(steps=2, learning_rate=1e-3))

    assert healed.training is False