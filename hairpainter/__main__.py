"""Entry point: python -m hairpainter"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow CLI batch mode: python -m hairpainter --input <dir> --output <dir> --batch
def _run_cli(args: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hair Painter — Fibril Segmentation")
    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", required=True, help="Output folder")
    parser.add_argument("--batch", action="store_true", help="Process all images in folder")
    parser.add_argument("--threshold", type=float, default=0.05, help="Frangi threshold (0-1)")
    parser.add_argument("--min-px", type=int, default=15, help="Minimum fibril length in pixels")
    parser.add_argument("--sam2", action="store_true", help="Enable SAM2 refinement")
    parsed = parser.parse_args(args)

    from hairpainter.orchestrator.pipeline import Pipeline, PipelineConfig
    from hairpainter.utils.types import PipelineInput

    config = PipelineConfig(
        frangi_threshold=parsed.threshold,
        min_fibril_px=parsed.min_px,
        use_sam2=parsed.sam2,
    )
    pipeline = Pipeline(config)
    output_dir = Path(parsed.output)
    input_path = Path(parsed.input)

    if parsed.batch or input_path.is_dir():
        extensions = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        paths = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]
    else:
        paths = [input_path]

    inputs = [PipelineInput(image_path=p, output_dir=output_dir) for p in paths]

    def _progress(step: str, pct: int) -> None:
        print(f"\r[{pct:3d}%] {step}", end="", flush=True)

    results = pipeline.run_batch(inputs, _progress)
    print()

    for r in results:
        status = "OK" if r.success else f"ERRO: {r.error}"
        fibril_info = ""
        if r.success and r.measure:
            m = r.measure
            fibril_info = (
                f" | Fibrilas: {len(m.fibrils)}"
                f" | Min: {m.min_nm:.1f}nm"
                f" | Média: {m.mean_nm:.1f}nm"
                f" | Máx: {m.max_nm:.1f}nm"
            )
        print(f"  {r.input_path.name}: {status}{fibril_info}")


def main() -> None:
    # Detect CLI vs GUI mode
    cli_flags = {"--input", "-i", "--batch", "--output", "-o", "--help", "-h"}
    if any(arg in cli_flags for arg in sys.argv[1:]):
        _run_cli(sys.argv[1:])
        return

    # GUI mode
    from PyQt6.QtWidgets import QApplication
    from hairpainter.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Hair Painter")
    app.setOrganizationName("Virologia UFMG")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
