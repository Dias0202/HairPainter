"""ScaleService — detect scale bar and compute pixels-per-nanometer ratio."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from hairpainter.utils.types import ImageData, ScaleResult

# Regex patterns for common TEM scale bar texts
_SCALE_PATTERN = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(nm|um|µm|μm|mm|Å|A)",
    re.IGNORECASE,
)

_UNIT_TO_NM: dict[str, float] = {
    "nm": 1.0,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "mm": 1_000_000.0,
    "å": 0.1,
    "a": 0.1,
}

# Tecnai TIFF pixel size tag — value is in Ångströms × 10^4 (empirically calibrated)
_TECNAI_ANGSTROM_FACTOR = 1e4


class ScaleService:
    def __init__(self, crop_fraction: float = 0.15, dark_threshold: int = 40) -> None:
        self._crop_fraction = crop_fraction
        self._dark_threshold = dark_threshold

    def detect(self, image_data: ImageData) -> ScaleResult:
        result = self._detect_visual(image_data)
        if result is not None:
            return result

        result = self._detect_from_metadata(image_data)
        if result is not None:
            return result

        # Fallback with placeholder — GUI will request manual entry
        return ScaleResult(
            px_per_nm=0.0,
            bar_bbox=(0, 0, 0, 0),
            scale_text="",
            scale_nm=0.0,
            source="manual",
            confidence=0.0,
        )

    # ------------------------------------------------------------------
    def _detect_visual(self, image_data: ImageData) -> ScaleResult | None:
        gray = image_data.array
        h, w = gray.shape

        # Crop bottom crop_fraction of image
        crop_y = int(h * (1.0 - self._crop_fraction))
        roi = gray[crop_y:, :]

        # Find dark region (scale bar background)
        dark_mask = roi < self._dark_threshold

        # Find horizontal white bar within dark region
        bar_info = self._find_white_bar(roi, dark_mask)
        if bar_info is None:
            return None
        bar_x, bar_y_in_roi, bar_w = bar_info

        # OCR the entire ROI to find scale text
        scale_text, scale_nm = self._ocr_scale(roi)
        if scale_nm is None:
            return None

        bar_y = crop_y + bar_y_in_roi
        px_per_nm = bar_w / scale_nm

        return ScaleResult(
            px_per_nm=px_per_nm,
            bar_bbox=(bar_x, bar_y, bar_w, 5),
            scale_text=scale_text,
            scale_nm=scale_nm,
            source="visual",
            confidence=0.9,
        )

    def _find_white_bar(
        self, roi: np.ndarray, dark_mask: np.ndarray
    ) -> tuple[int, int, int] | None:
        """Return (x, y, width) of the longest horizontal white segment inside dark_mask."""
        h, w = roi.shape
        best = (0, 0, 0)  # x, y, width

        for y in range(h):
            if not dark_mask[y].any():
                continue
            row = roi[y].astype(np.int32)
            # Find white pixels (> 200) within dark rows
            white = (row > 200).astype(np.uint8)
            segments = self._run_lengths(white)
            for x_start, length in segments:
                if length > best[2]:
                    best = (x_start, y, length)

        if best[2] < 10:  # bar must be at least 10px wide
            return None
        return best

    @staticmethod
    def _run_lengths(arr: np.ndarray) -> list[tuple[int, int]]:
        """Return list of (start_x, length) for runs of 1s in 1D binary array."""
        result = []
        in_run = False
        start = 0
        for i, v in enumerate(arr):
            if v and not in_run:
                in_run = True
                start = i
            elif not v and in_run:
                in_run = False
                result.append((start, i - start))
        if in_run:
            result.append((start, len(arr) - start))
        return result

    def _ocr_scale(self, roi: np.ndarray) -> tuple[str, float | None]:
        try:
            import easyocr
        except ImportError:
            return "", None

        # EasyOCR reader is expensive to init; cache on instance
        if not hasattr(self, "_reader"):
            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        # Invert ROI for better OCR on white-on-black text
        inverted = cv2.bitwise_not(roi)
        results = self._reader.readtext(inverted, detail=0, paragraph=False)
        text_blob = " ".join(results)

        match = _SCALE_PATTERN.search(text_blob)
        if not match:
            return text_blob, None

        value_str = match.group(1).replace(",", ".")
        unit = match.group(2).lower()
        value_nm = float(value_str) * _UNIT_TO_NM.get(unit, 1.0)
        return f"{value_str} {match.group(2)}", value_nm

    # ------------------------------------------------------------------
    def _detect_from_metadata(self, image_data: ImageData) -> ScaleResult | None:
        raw = image_data.metadata.get("pixel_x_raw")
        if raw is None:
            return None

        try:
            raw_val = float(raw)
        except (TypeError, ValueError):
            return None

        # Tecnai stores pixel size in Å × factor; empirical: divide by 1e4 → Å → ×0.1 → nm
        pixel_size_nm = (raw_val / _TECNAI_ANGSTROM_FACTOR) * 0.1

        if pixel_size_nm <= 0:
            return None

        px_per_nm = 1.0 / pixel_size_nm

        return ScaleResult(
            px_per_nm=px_per_nm,
            bar_bbox=(0, 0, 0, 0),
            scale_text=f"metadata ({pixel_size_nm:.3f} nm/px)",
            scale_nm=pixel_size_nm,
            source="metadata",
            confidence=0.6,
        )
