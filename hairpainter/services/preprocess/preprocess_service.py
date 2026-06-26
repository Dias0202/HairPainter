"""PreprocessService — normalize and enhance TEM images for fibril detection."""
from __future__ import annotations

import cv2
import numpy as np

from hairpainter.utils.types import ImageData


class PreprocessService:
    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8) -> None:
        self._clip_limit = clip_limit
        self._tile_grid = tile_grid

    def enhance(self, image_data: ImageData) -> np.ndarray:
        """
        Return enhanced uint8 grayscale array ready for segmentation.

        Steps:
        1. Normalize to full [0, 255] range
        2. Histogram equalization if image is very dark or flat
        3. CLAHE for local contrast enhancement
        """
        gray = image_data.array.copy()

        # 1. Stretch histogram to full range
        gray = self._normalize(gray)

        # 2. If dynamic range is very compressed, apply global equalization first
        if gray.std() < 20:
            gray = cv2.equalizeHist(gray)

        # 3. CLAHE — preserves local contrast, critical for thin fibrils
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=(self._tile_grid, self._tile_grid),
        )
        return clahe.apply(gray)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi == lo:
            return arr
        return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
