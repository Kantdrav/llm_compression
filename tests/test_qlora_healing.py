from __future__ import annotations

import torch
from torch import nn

from llm_edge_compression.qlora_healing import find_lora_target_modules


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.q_proj = nn.Linear(8, 8)
        self.model.v_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 8)

    def forward(self, input_ids):
        return self.lm_head(self.model.v_proj(self.model.q_proj(input_ids.float())))


def test_lora_target_detection_ignores_lm_head():
    model = TinyCausalLM()
    assert find_lora_target_modules(model) == ["q_proj", "v_proj"]
