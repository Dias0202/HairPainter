from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class PipelineInput:
    image_path: Path
    output_dir: Path
    use_sam2: bool = False
    min_fibril_px: int = 15
    frangi_threshold: float = 0.3


@dataclass
class ImageData:
    array: np.ndarray           # uint8, grayscale 2D
    original_array: np.ndarray  # uint8, preserved for rendering (may be RGB)
    metadata: dict
    source_path: Path


@dataclass
class ScaleResult:
    px_per_nm: float
    bar_bbox: tuple[int, int, int, int]  # x, y, w, h in original image coords
    scale_text: str                       # e.g. "500 nm"
    scale_nm: float                       # numeric value in nm
    source: Literal["visual", "metadata", "manual"]
    confidence: float = 1.0              # 0.0–1.0


@dataclass
class CapsidResult:
    center: tuple[int, int]   # (x, y) pixel
    radius: int               # pixels
    mask: np.ndarray          # bool, True = capsid region


@dataclass
class FibrilInstance:
    id: int
    mask: np.ndarray          # bool, same HxW as image
    skeleton: np.ndarray      # bool, thinned 1-px path
    length_px: float
    length_nm: float = 0.0


@dataclass
class SegmentResult:
    label_map: np.ndarray     # int32, 0 = background, 1..N = fibril ids
    fibrils: list[FibrilInstance] = field(default_factory=list)

    @property
    def n_fibrils(self) -> int:
        return len(self.fibrils)


@dataclass
class MeasureResult:
    fibrils: list[FibrilInstance]
    min_nm: float
    mean_nm: float
    max_nm: float
    std_nm: float
    histogram: dict = field(default_factory=dict)  # {"bins": [...], "counts": [...]}


@dataclass
class RenderResult:
    fibrils_only_path: Path   # Deliverable 1
    overlay_path: Path        # Deliverable 2
    measured_path: Path       # Deliverable 3
    report_path: Path         # JSON


@dataclass
class PipelineResult:
    input_path: Path
    scale: ScaleResult | None = None
    capsid: CapsidResult | None = None
    segment: SegmentResult | None = None
    measure: MeasureResult | None = None
    render: RenderResult | None = None
    success: bool = False
    error: str = ""
