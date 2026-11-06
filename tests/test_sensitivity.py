import json

import torch
from torch import nn

from llm_edge_compression.config import CompressionPolicy
from llm_edge_compression.sensitivity import profile_mpo_sensitivity, write_sensitivity_profile


class SensitivityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.protected = nn.Linear(16, 16)
        self.compressible = nn.Linear(16, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.compressible(torch.relu(self.protected(inputs)))


def test_profile_measures_only_allowed_layers_and_writes_json(tmp_path) -> None:
    torch.manual_seed(0)
    model = SensitivityModel().eval()
    batches = [torch.randn(2, 16) for _ in range(2)]

    results = profile_mpo_sensitivity(
        model,
        batches,
        bond_dims=(1, 8),
        mpo_sites=2,
        layer_policy=CompressionPolicy(skip_module_patterns=(r"^protected$",)),
    )

    assert [result.module_path for result in results] == ["compressible", "compressible"]
    assert [result.bond_dim for result in results] == [1, 8]
    assert results[0].compressed_parameters < results[0].original_parameters
    assert results[0].output_mse >= 0
    assert results[1].output_mse <= results[0].output_mse + 1e-6

    output = write_sensitivity_profile(results, tmp_path / "profile.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload[0]["module_path"] == "compressible"
    assert payload[0]["bond_dim"] == 1
