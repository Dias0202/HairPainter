"""IOService — image loading with TIFF stack support and metadata extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hairpainter.utils.types import ImageData

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# TIFF tag IDs from Tecnai/FEI microscopes
_TAG_PIXEL_X = 65450
_TAG_PIXEL_Y = 65451


class IOService:
    def load(self, path: Path) -> ImageData:
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {suffix}. Use {SUPPORTED_EXTENSIONS}")

        if suffix in {".tif", ".tiff"}:
            return self._load_tiff(path)
        return self._load_standard(path)

    # ------------------------------------------------------------------
    def _load_tiff(self, path: Path) -> ImageData:
        import tifffile

        metadata: dict = {}

        with tifffile.TiffFile(str(path)) as tif:
            # Extract all frames from stack
            frames: list[np.ndarray] = []
            for page in tif.pages:
                frames.append(page.asarray())

            # Select frame with highest variance (most informative)
            best_frame = max(frames, key=lambda f: float(np.var(f)))

            # Extract Tecnai pixel-size metadata
            for page in tif.pages[:1]:
                tags = page.tags
                if _TAG_PIXEL_X in tags:
                    metadata["pixel_x_raw"] = tags[_TAG_PIXEL_X].value
                if _TAG_PIXEL_Y in tags:
                    metadata["pixel_y_raw"] = tags[_TAG_PIXEL_Y].value
                # Also try to grab XML metadata (tag 34682)
                if 34682 in tags:
                    metadata["microscope_xml"] = tags[34682].value

            metadata["n_frames"] = len(frames)
            metadata["selected_frame_variance"] = float(np.var(best_frame))

        array = self._to_uint8_gray(best_frame)
        original = self._to_rgb(best_frame)
        return ImageData(array=array, original_array=original, metadata=metadata, source_path=path)

    def _load_standard(self, path: Path) -> ImageData:
        from PIL import Image

        img = Image.open(path)
        metadata: dict = {"format": img.format, "mode": img.mode}

        original = np.array(img.convert("RGB"), dtype=np.uint8)
        array = self._to_uint8_gray(original)
        return ImageData(array=array, original_array=original, metadata=metadata, source_path=path)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_uint8_gray(arr: np.ndarray) -> np.ndarray:
        """Convert any array to uint8 grayscale 2D."""
        if arr.ndim == 3:
            # RGB/RGBA → grayscale via luminosity
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        elif arr.ndim == 2:
            gray = arr.astype(np.float64)
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

        if gray.max() > 255:
            # 16-bit → 8-bit
            gray = (gray / gray.max() * 255)
        return gray.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _to_rgb(arr: np.ndarray) -> np.ndarray:
        """Convert any array to uint8 RGB 3-channel."""
        from PIL import Image

        if arr.dtype != np.uint8:
            arr = (arr / arr.max() * 255).clip(0, 255).astype(np.uint8)

        if arr.ndim == 2:
            img = Image.fromarray(arr, mode="L").convert("RGB")
            return np.array(img, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA").convert("RGB")
            return np.array(img, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr.astype(np.uint8)

        # Paletted: convert via PIL
        img = Image.fromarray(arr).convert("RGB")
        return np.array(img, dtype=np.uint8)
