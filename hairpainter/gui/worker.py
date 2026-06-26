"""PipelineWorker — runs the pipeline in a background QThread."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from hairpainter.orchestrator.pipeline import Pipeline, PipelineConfig
from hairpainter.utils.types import PipelineInput, PipelineResult


class PipelineWorker(QThread):
    progress = pyqtSignal(str, int)           # (step_name, percent 0-100)
    image_done = pyqtSignal(object)           # PipelineResult (typed as object for Qt)
    all_done = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(
        self,
        image_paths: list[Path],
        output_dir: Path,
        config: PipelineConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._paths = image_paths
        self._output_dir = output_dir
        self._config = config
        self._pipeline = Pipeline(config)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        results: list[PipelineResult] = []
        total = len(self._paths)

        for i, path in enumerate(self._paths):
            if self._cancelled:
                break

            def _progress(step: str, pct: int, i=i, total=total) -> None:
                global_pct = int((i * 100 + pct) / total)
                self.progress.emit(f"[{i + 1}/{total}] {step}", global_pct)

            inp = PipelineInput(
                image_path=path,
                output_dir=self._output_dir,
                use_sam2=self._config.use_sam2,
                min_fibril_px=self._config.min_fibril_px,
                frangi_threshold=self._config.frangi_threshold,
            )
            result = self._pipeline.run(inp, _progress)
            results.append(result)
            self.image_done.emit(result)

        self.all_done.emit(results)
