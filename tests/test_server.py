from pathlib import Path

from fastapi.testclient import TestClient

from llm_edge_compression import server


client = TestClient(server.app)


def test_compress_remote_routes_through_download_and_bundle(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(work_dir: Path, model_id: str, revision: str | None = None) -> Path:
        captured["download"] = (work_dir, model_id, revision)
        return work_dir / "downloaded-model"

    def fake_compress_model_bundle(
        source_model_dir: Path,
        model_id: str,
        method: str,
        rank_ratio: float,
        target_device: str,
        heal_steps: int,
        heal_learning_rate: float,
        calibration_batches: int,
        calibration_batch_size: int,
        calibration_sequence_length: int,
        output_root: str,
        trust_remote_code: bool,
    ) -> dict[str, object]:
        captured["compress"] = {
            "source_model_dir": source_model_dir,
            "model_id": model_id,
            "method": method,
            "rank_ratio": rank_ratio,
            "target_device": target_device,
            "heal_steps": heal_steps,
            "heal_learning_rate": heal_learning_rate,
            "calibration_batches": calibration_batches,
            "calibration_batch_size": calibration_batch_size,
            "calibration_sequence_length": calibration_sequence_length,
            "output_root": output_root,
            "trust_remote_code": trust_remote_code,
        }
        return {"manifest": "bundle/manifest.json", "output_dir": "bundle", "parameter_ratio": 0.5, "metrics": {}, "model_id": model_id}

    monkeypatch.setattr(server, "_download_remote_model", fake_download)
    monkeypatch.setattr(server, "_compress_model_bundle", fake_compress_model_bundle)

    response = client.post(
        "/compress-remote",
        json={
            "model_id": "gpt2",
            "revision": "main",
            "method": "quantize",
            "rank_ratio": 0.25,
            "target_device": "cpu",
            "trust_remote_code": True,
            "heal_steps": 2,
            "heal_learning_rate": 0.0002,
            "calibration_batches": 3,
            "calibration_batch_size": 4,
            "calibration_sequence_length": 8,
            "output_root": "artifacts/test-uploads",
        },
    )

    assert response.status_code == 200
    assert response.json()["model_id"] == "gpt2"
    assert captured["download"][1] == "gpt2"
    assert captured["download"][2] == "main"
    assert captured["compress"]["trust_remote_code"] is True
    assert captured["compress"]["method"] == "quantize"
