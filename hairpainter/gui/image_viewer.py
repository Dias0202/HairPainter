"""ImageViewer — zoomable, pannable image display widget."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class ImageViewer(QWidget):
    def __init__(self, title: str = "", parent=None) -> None:
        super().__init__(parent)
        self._scale = 1.0
        self._pixmap: QPixmap | None = None

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._label.setMinimumSize(200, 150)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self._label)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if title:
            from PyQt6.QtWidgets import QLabel as QTitleLabel
            title_lbl = QTitleLabel(title)
            title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_lbl.setStyleSheet("font-weight: bold; color: #1c3052;")
            layout.addWidget(title_lbl)

        layout.addWidget(self._scroll)

    def load_path(self, path: Path) -> None:
        px = QPixmap(str(path))
        if px.isNull():
            return
        self._pixmap = px
        self._scale = 1.0
        self._update_display()

    def clear(self) -> None:
        self._pixmap = None
        self._label.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._scale = max(0.1, min(10.0, self._scale * factor))
        self._update_display()

    def _update_display(self) -> None:
        if self._pixmap is None:
            return
        w = int(self._pixmap.width() * self._scale)
        h = int(self._pixmap.height() * self._scale)
        scaled = self._pixmap.scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
