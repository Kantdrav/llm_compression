from pathlib import Path

from llm_edge_compression.manifest import ModelManifest, read_manifest, write_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ModelManifest(
        model_id="test-model",
        method="tensor_inspired",
        artifact_files=["compressed_model.pt"],
        metrics={"original_parameters": 10, "compressed_parameters": 5},
    )

    path = write_manifest(manifest, tmp_path)
    restored = read_manifest(path)

    assert restored.model_id == manifest.model_id
    assert restored.method == manifest.method
    assert restored.artifact_files == manifest.artifact_files
    assert restored.metrics == manifest.metrics
