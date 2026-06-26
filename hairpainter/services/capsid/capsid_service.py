"""CapsidService — detect the central viral capsid to exclude it from fibril segmentation."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from hairpainter.utils.types import CapsidResult

# Mimivirus at 98 000× mag, ~2 nm/px → diameter ~700 nm ≈ 350 px → radius ≈ 175 px.
# Allow some variation across images: search r=80..250 from image center.
_MIN_CAPSID_R = 80
_MAX_CAPSID_R = 250
_DEFAULT_CAPSID_R = 175

# Brightness threshold: below this value a ring is considered part of the capsid
_BRIGHTNESS_CAPSID_THRESH = 115


class CapsidService:
    def __init__(
        self,
        min_radius: int = _MIN_CAPSID_R,
        max_radius: int = _MAX_CAPSID_R,
        default_radius: int = _DEFAULT_CAPSID_R,
        brightness_thresh: int = _BRIGHTNESS_CAPSID_THRESH,
    ) -> None:
        self._min_r = min_radius
        self._max_r = max_radius
        self._default_r = default_radius
        self._brightness_thresh = brightness_thresh

    def detect(self, enhanced: np.ndarray) -> CapsidResult:
        h, w = enhanced.shape

        # Step 1: find capsid center (centroid of darkest central region)
        center = self._find_center(enhanced, h, w)

        # Step 2: find capsid radius from radial brightness profile
        radius = self._radial_radius(enhanced, h, w, center)

        mask = self._make_mask(h, w, center, radius)
        return CapsidResult(center=center, radius=radius, mask=mask)

    # ------------------------------------------------------------------
    def _find_center(self, gray: np.ndarray, h: int, w: int) -> tuple[int, int]:
        """
        Find the capsid center as the centroid of the darkest pixels
        within the central 50% of the image.
        """
        # Only look in the central region to avoid being misled by scale bar
        cy0, cy1 = h // 4, 3 * h // 4
        cx0, cx1 = w // 4, 3 * w // 4
        roi = gray[cy0:cy1, cx0:cx1]

        # Heavy blur to suppress fibril texture, keep bulk capsid
        blurred = cv2.GaussianBlur(roi, (51, 51), 15)

        # Threshold: keep darkest 20% of pixels in the ROI
        thresh_val = int(np.percentile(blurred, 20))
        dark_mask = blurred <= thresh_val

        if dark_mask.sum() < 100:
            return (w // 2, h // 2)

        # Centroid of dark region
        ys, xs = np.where(dark_mask)
        cx_roi = int(xs.mean()) + cx0
        cy_roi = int(ys.mean()) + cy0
        return (cx_roi, cy_roi)

    def _radial_radius(
        self, gray: np.ndarray, h: int, w: int, center: tuple[int, int]
    ) -> int:
        """
        Find capsid radius from radial brightness profile using an ADAPTIVE
        threshold: the midpoint between the capsid interior brightness and
        the distant background brightness.  This is more robust than a fixed
        absolute threshold across images with different global contrast.
        """
        cx, cy = center
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        scale_bar_y = int(h * 0.85)
        valid = np.ones((h, w), dtype=bool)
        valid[scale_bar_y:, :] = False

        step = 5

        # Build a radial brightness profile from r=150 to max_r.
        # Search only from 150 px because Mimivirus capsid at 98,000× is always
        # larger than that; starting here avoids spurious early peaks from
        # capsid-wall bright rings or nearby contamination particles.
        r_start = max(self._min_r, 150)
        profile: list[float] = []
        r_samples: list[int] = []
        for r in range(r_start, self._max_r + 1, step):
            ring = (dist >= r) & (dist < r + step) & valid
            if ring.sum() < 20:
                continue
            profile.append(float(gray[ring].mean()))
            r_samples.append(r)

        if len(profile) < 4:
            return self._default_r

        # Smooth profile and find the radius of maximum upward gradient.
        # The capsid boundary is where brightness rises most steeply from
        # the dark interior to the bright background / sparse fibril region.
        prof_arr = np.array(profile)
        # 3-point moving average to suppress per-ring noise
        smoothed = np.convolve(prof_arr, np.ones(3) / 3, mode="valid")
        r_arr = np.array(r_samples[1: len(r_samples) - 1])  # matches convolution valid length
        grad = np.gradient(smoothed)
        peak_idx = int(np.argmax(grad))
        return int(r_arr[peak_idx])

    @staticmethod
    def _make_mask(h: int, w: int, center: tuple[int, int], radius: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        return mask.astype(bool)
