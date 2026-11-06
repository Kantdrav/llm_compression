import torch
from torch import nn

from llm_edge_compression.config import CompressionConfig
from llm_edge_compression.pipeline import CompressionPipeline


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(8, 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


def test_pipeline_snapshots_teacher_before_in_place_compression(monkeypatch, tmp_path) -> None:
    original = TinyModel()
    pipeline = CompressionPipeline(
        CompressionConfig(model_id="demo", output_dir=tmp_path, heal_steps=1),
    )
    captured: dict[str, nn.Module] = {}

    monkeypatch.setattr(
        "llm_edge_compression.pipeline.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: original,
    )
    monkeypatch.setattr("llm_edge_compression.pipeline._state_dict_size_bytes", lambda model: 1)
    monkeypatch.setattr(pipeline, "_build_calibration_batches", lambda model: iter([torch.randn(2, 8)]))

    class IdentityCompressor:
        def compress(self, model: nn.Module) -> nn.Module:
            captured["compressed"] = model
            model.layer = nn.Linear(8, 8)
            return model

    monkeypatch.setattr(pipeline, "_build_compressor", lambda: IdentityCompressor())

    def fake_heal(student, teacher, batches, config):
        captured["teacher"] = teacher
        return student

    monkeypatch.setattr("llm_edge_compression.pipeline.heal_model", fake_heal)
    monkeypatch.setattr(
        "llm_edge_compression.pipeline.EdgeExporter.export_bundle",
        lambda self, model: type("ExportResult", (), {"artifact_files": []})(),
    )

    pipeline.run()

    assert captured["teacher"] is not captured["compressed"]
    assert isinstance(captured["teacher"].layer, nn.Linear)
