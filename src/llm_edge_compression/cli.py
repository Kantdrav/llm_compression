from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import CompressionConfig, ExportConfig
from .demo import run_demo
from .pipeline import CompressionPipeline

app = typer.Typer(add_completion=False, help="Compress LLMs and export edge-ready bundles.")


@app.command()
def compress(
    model_id: str = typer.Option(..., help="Model name or local checkpoint path."),
    output_dir: Path = typer.Option(..., help="Directory for compressed artifacts."),
    method: str = typer.Option("tensor_inspired", help="Compression method: quantize or tensor_inspired."),
    rank_ratio: float = typer.Option(0.5, help="Fraction of the maximum linear-layer rank to keep."),
    target_device: str = typer.Option("cpu", help="Target deployment device label."),
    heal_steps: int = typer.Option(0, help="Optional healing steps to run after compression."),
    heal_learning_rate: float = typer.Option(1e-4, help="Learning rate for healing.") ,
    calibration_batches: int = typer.Option(8, help="Number of synthetic calibration batches to use for healing."),
    calibration_batch_size: int = typer.Option(2, help="Synthetic calibration batch size."),
    calibration_sequence_length: int = typer.Option(16, help="Synthetic calibration sequence length."),
) -> None:
    compression = CompressionConfig(
        model_id=model_id,
        output_dir=output_dir,
        method=method,
        rank_ratio=rank_ratio,
        target_device=target_device,
        heal_steps=heal_steps,
        heal_learning_rate=heal_learning_rate,
        calibration_batches=calibration_batches,
        calibration_batch_size=calibration_batch_size,
        calibration_sequence_length=calibration_sequence_length,
    )
    export = ExportConfig(output_dir=output_dir)
    result = CompressionPipeline(compression, export).run()
    typer.echo(json.dumps({"manifest": result.manifest_path.as_posix(), "metrics": result.manifest.metrics}, indent=2))


@app.command()
def inspect(output_dir: Path = typer.Option(..., help="Directory containing a manifest.json file.")) -> None:
    manifest_path = output_dir / "manifest.json"
    typer.echo(manifest_path.read_text(encoding="utf-8"))


@app.command()
def demo(
    output_dir: Path = typer.Option(Path("artifacts/demo"), help="Directory for the demo bundle."),
    rank_ratio: float = typer.Option(0.5, help="Fraction of the maximum linear-layer rank to keep."),
) -> None:
    result = run_demo(output_dir=output_dir, rank_ratio=rank_ratio)
    typer.echo(
        json.dumps(
            {
                "manifest": result.manifest_path.as_posix(),
                "output_dir": result.output_dir.as_posix(),
                "parameter_ratio": round(result.parameter_ratio, 6),
            },
            indent=2,
        )
    )

def main() -> None:
    app()


if __name__ == "__main__":
    main()
