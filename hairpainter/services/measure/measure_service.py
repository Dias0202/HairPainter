"""MeasureService — compute real-world fibril lengths from pixel skeletons."""
from __future__ import annotations

import numpy as np

from hairpainter.utils.types import FibrilInstance, MeasureResult, ScaleResult, SegmentResult


class MeasureService:
    def measure(self, segment: SegmentResult, scale: ScaleResult) -> MeasureResult:
        if scale.px_per_nm <= 0:
            raise ValueError(
                "Scale not calibrated (px_per_nm = 0). "
                "Provide a valid scale bar or enter scale manually."
            )

        fibrils: list[FibrilInstance] = []
        for f in segment.fibrils:
            # length_px is already computed and stored in FibrilInstance by SegmentService.
            # If somehow it's 0 (edge case), recompute from skeleton.
            length_px = f.length_px if f.length_px > 0 else float(f.skeleton.sum())
            length_nm = length_px / scale.px_per_nm

            fibrils.append(
                FibrilInstance(
                    id=f.id,
                    mask=f.mask,
                    skeleton=f.skeleton,
                    length_px=length_px,
                    length_nm=length_nm,
                )
            )

        # Filter out zero-length entries (should not occur, but be safe)
        fibrils = [f for f in fibrils if f.length_nm > 0]

        if not fibrils:
            return MeasureResult(
                fibrils=[],
                min_nm=0.0,
                mean_nm=0.0,
                max_nm=0.0,
                std_nm=0.0,
                histogram={},
            )

        lengths = np.array([f.length_nm for f in fibrils])
        histogram = self._build_histogram(lengths)

        return MeasureResult(
            fibrils=fibrils,
            min_nm=float(lengths.min()),
            mean_nm=float(lengths.mean()),
            max_nm=float(lengths.max()),
            std_nm=float(lengths.std()),
            histogram=histogram,
        )

    @staticmethod
    def _build_histogram(lengths: np.ndarray, bins: int = 20) -> dict:
        counts, bin_edges = np.histogram(lengths, bins=bins)
        return {
            "bins": [round(float(e), 2) for e in bin_edges.tolist()],
            "counts": counts.tolist(),
        }
