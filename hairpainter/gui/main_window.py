"""MainWindow — primary GUI window for Hair Painter."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from hairpainter.gui.image_viewer import ImageViewer
from hairpainter.gui.worker import PipelineWorker
from hairpainter.orchestrator.pipeline import PipelineConfig
from hairpainter.utils.types import PipelineResult

NAVY = "#1c3052"
NAVY_LIGHT = "#2d4a7a"
WHITE = "#ffffff"
LIGHT_GRAY = "#f0f2f5"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hair Painter v1.0")
        self.setMinimumSize(1100, 700)

        self._image_paths: list[Path] = []
        self._output_dir: Path = Path.home() / "HairPainter_output"
        self._worker: PipelineWorker | None = None
        self._current_result: PipelineResult | None = None

        self._build_ui()
        self._apply_style()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel: input + settings
        left = QWidget()
        left.setFixedWidth(280)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_input_group())
        left_layout.addWidget(self._build_settings_group())
        left_layout.addStretch()
        left_layout.addWidget(self._build_action_group())

        main_layout.addWidget(left)

        # Right panel: previews + stats
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._build_preview_group())
        right_layout.addWidget(self._build_stats_bar())

        main_layout.addWidget(right, 1)

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Pronto. Selecione imagens para começar.")

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Entrada")
        layout = QVBoxLayout(group)

        btn_add = QPushButton("+ Adicionar Imagens")
        btn_add.clicked.connect(self._on_add_images)
        layout.addWidget(btn_add)

        self._file_list = QListWidget()
        self._file_list.setMaximumHeight(180)
        layout.addWidget(self._file_list)

        btn_out = QPushButton("Selecionar Pasta de Saída")
        btn_out.clicked.connect(self._on_select_output)
        layout.addWidget(btn_out)

        self._lbl_output = QLabel(str(self._output_dir))
        self._lbl_output.setWordWrap(True)
        self._lbl_output.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._lbl_output)

        return group

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Configurações")
        layout = QVBoxLayout(group)

        self._chk_sam2 = QCheckBox("Usar SAM2 (mais lento, mais preciso)")
        layout.addWidget(self._chk_sam2)

        layout.addWidget(QLabel("Threshold Frangi:"))
        self._slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self._slider_threshold.setRange(5, 80)
        self._slider_threshold.setValue(30)
        self._lbl_threshold = QLabel("0.30")
        self._slider_threshold.valueChanged.connect(
            lambda v: self._lbl_threshold.setText(f"{v / 100:.2f}")
        )
        h = QHBoxLayout()
        h.addWidget(self._slider_threshold)
        h.addWidget(self._lbl_threshold)
        layout.addLayout(h)

        layout.addWidget(QLabel("Fibrilas mínimas (px):"))
        self._spin_min_px = QSpinBox()
        self._spin_min_px.setRange(5, 100)
        self._spin_min_px.setValue(15)
        layout.addWidget(self._spin_min_px)

        return group

    def _build_action_group(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._btn_process = QPushButton("PROCESSAR")
        self._btn_process.setMinimumHeight(42)
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._on_process)
        layout.addWidget(self._btn_process)

        self._btn_cancel = QPushButton("Cancelar")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._on_cancel)
        layout.addWidget(self._btn_cancel)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._lbl_step = QLabel("")
        self._lbl_step.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(self._lbl_step)

        return container

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Entregáveis")
        layout = QHBoxLayout(group)

        self._viewer1 = ImageViewer("1 — Fibrilas (fundo preto)")
        self._viewer2 = ImageViewer("2 — Overlay")
        self._viewer3 = ImageViewer("3 — Overlay + Medidas")

        for v in (self._viewer1, self._viewer2, self._viewer3):
            layout.addWidget(v, 1)

        return group

    def _build_stats_bar(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)

        self._lbl_fibril_count = QLabel("Fibrilas: —")
        self._lbl_min = QLabel("Min: —")
        self._lbl_mean = QLabel("Média: —")
        self._lbl_max = QLabel("Máx: —")
        self._btn_export = QPushButton("Exportar Relatório JSON")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export)

        for w in (self._lbl_fibril_count, self._lbl_min, self._lbl_mean, self._lbl_max):
            layout.addWidget(w)
            layout.addWidget(self._make_sep())
        layout.addStretch()
        layout.addWidget(self._btn_export)
        return container

    @staticmethod
    def _make_sep() -> QLabel:
        sep = QLabel("|")
        sep.setStyleSheet("color: lightgray;")
        return sep

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    def _on_add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar Imagens MET",
            str(Path.home()),
            "Imagens (*.tif *.tiff *.jpg *.jpeg *.png)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._image_paths:
                self._image_paths.append(path)
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self._file_list.addItem(item)

        self._btn_process.setEnabled(bool(self._image_paths))

    def _on_select_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Pasta de Saída", str(Path.home()))
        if folder:
            self._output_dir = Path(folder)
            self._lbl_output.setText(folder)

    def _on_process(self) -> None:
        if not self._image_paths:
            return

        config = PipelineConfig(
            frangi_threshold=self._slider_threshold.value() / 100.0,
            min_fibril_px=self._spin_min_px.value(),
            use_sam2=self._chk_sam2.isChecked(),
        )

        self._worker = PipelineWorker(
            image_paths=self._image_paths,
            output_dir=self._output_dir,
            config=config,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)

        self._btn_process.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._progress_bar.setValue(0)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self._worker.wait(3000)
        self._reset_controls()
        self.statusBar().showMessage("Cancelado pelo usuário.")

    def _on_progress(self, step: str, pct: int) -> None:
        self._progress_bar.setValue(pct)
        self._lbl_step.setText(step)
        self.statusBar().showMessage(step)

    def _on_image_done(self, result: PipelineResult) -> None:
        self._current_result = result

        if not result.success:
            self.statusBar().showMessage(f"Erro em {result.input_path.name}: {result.error}")
            return

        # Update previews
        if result.render:
            self._viewer1.load_path(result.render.fibrils_only_path)
            self._viewer2.load_path(result.render.overlay_path)
            self._viewer3.load_path(result.render.measured_path)

        # Update stats
        if result.measure:
            m = result.measure
            self._lbl_fibril_count.setText(f"Fibrilas: {len(m.fibrils)}")
            self._lbl_min.setText(f"Min: {m.min_nm:.1f} nm")
            self._lbl_mean.setText(f"Média: {m.mean_nm:.1f} nm")
            self._lbl_max.setText(f"Máx: {m.max_nm:.1f} nm")
            self._btn_export.setEnabled(True)

    def _on_all_done(self, results: list) -> None:
        self._reset_controls()
        ok = sum(1 for r in results if r.success)
        self.statusBar().showMessage(
            f"Processamento concluído: {ok}/{len(results)} imagens com sucesso."
        )

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Erro no Pipeline", msg)
        self._reset_controls()

    def _on_export(self) -> None:
        if self._current_result and self._current_result.render:
            import subprocess
            import sys
            report = self._current_result.render.report_path
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(report)])
            else:
                subprocess.Popen(["xdg-open", str(report.parent)])

    def _reset_controls(self) -> None:
        self._btn_process.setEnabled(bool(self._image_paths))
        self._btn_cancel.setEnabled(False)

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    def _apply_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {LIGHT_GRAY}; }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #c0c8d8;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 10px;
                background: {WHITE};
            }}
            QGroupBox::title {{
                color: {NAVY};
                subcontrol-origin: margin;
                left: 8px;
            }}
            QPushButton {{
                background-color: {NAVY};
                color: {WHITE};
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {NAVY_LIGHT}; }}
            QPushButton:disabled {{ background-color: #a0aab8; }}
            QProgressBar {{
                border: 1px solid #c0c8d8;
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{ background-color: {NAVY}; border-radius: 3px; }}
        """)
