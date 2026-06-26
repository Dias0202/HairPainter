"""RenderService — generate the three deliverables and JSON report."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hairpainter.utils.color import (
    FIBRIL_COLOR_RGB,
    OVERLAY_ALPHA,
    apply_fibril_color,
    blend_overlay,
)
from hairpainter.utils.types import (
    CapsidResult,
    ImageData,
    MeasureResult,
    RenderResult,
    ScaleResult,
    SegmentResult,
)


class RenderService:
    def render(
        self,
        image_data: ImageData,
        segment: SegmentResult,
        measure: MeasureResult,
        scale: ScaleResult,
        capsid: CapsidResult,
        output_dir: Path,
    ) -> RenderResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = image_data.source_path.stem

        h, w = image_data.array.shape
        original_rgb = image_data.original_array  # H x W x 3

        # Build RGBA fibril layer (shared by all three deliverables)
        fibril_rgba, overlap_map = self._build_fibril_layer(h, w, segment)

        # --- Deliverable 1: black background + fibrils only ---
        d1_path = output_dir / f"{stem}_fibrils_only.png"
        d1 = self._render_fibrils_only(fibril_rgba)
        Image.fromarray(d1, "RGBA").save(str(d1_path))

        # --- Deliverable 2: raw + overlay ---
        d2_path = output_dir / f"{stem}_overlay.png"
        d2 = blend_overlay(original_rgb, fibril_rgba)
        Image.fromarray(d2, "RGB").save(str(d2_path))

        # --- Deliverable 3: overlay + measurements annotation ---
        d3_path = output_dir / f"{stem}_measured.png"
        d3 = self._render_measured(d2, measure, scale)
        Image.fromarray(d3, "RGB").save(str(d3_path))

        # --- JSON report ---
        report_path = output_dir / f"{stem}_report.json"
        self._write_report(
            report_path=report_path,
            image_data=image_data,
            scale=scale,
            capsid=capsid,
            measure=measure,
        )

        return RenderResult(
            fibrils_only_path=d1_path,
            overlay_path=d2_path,
            measured_path=d3_path,
            report_path=report_path,
        )

    # ------------------------------------------------------------------
    def _build_fibril_layer(
        self, h: int, w: int, segment: SegmentResult
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns RGBA fibril canvas and per-pixel overlap count."""
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        overlap_count = np.zeros((h, w), dtype=np.int32)

        for fibril in segment.fibrils:
            apply_fibril_color(canvas, fibril.mask, overlap_count)
            overlap_count[fibril.mask] += 1

        return canvas, overlap_count

    @staticmethod
    def _render_fibrils_only(fibril_rgba: np.ndarray) -> np.ndarray:
        """Black background (A=0) with fibril pixels fully composited."""
        out = fibril_rgba.copy()
        # Where no fibril: keep fully transparent (already zeros)
        return out

    @staticmethod
    def _render_measured(
        overlay_rgb: np.ndarray,
        measure: MeasureResult,
        scale: ScaleResult,
    ) -> np.ndarray:
        """Draw measurement annotation box on a copy of the overlay image."""
        img_pil = Image.fromarray(overlay_rgb, "RGB")
        draw = ImageDraw.Draw(img_pil)

        h, w = overlay_rgb.shape[:2]

        text_lines = [
            f"Fibrilas: {len(measure.fibrils)}",
            f"Min: {measure.min_nm:.1f} nm",
            f"Media: {measure.mean_nm:.1f} nm",
            f"Max: {measure.max_nm:.1f} nm",
            f"Escala: {scale.scale_text} ({scale.source})",
        ]

        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, h // 50))
        except OSError:
            font = ImageFont.load_default()

        # Box dimensions
        line_h = max(18, h // 45)
        box_w = max(220, w // 5)
        box_h = line_h * len(text_lines) + 16
        margin = 10
        bx, by = margin, h - box_h - margin

        # Semi-transparent dark background
        overlay = img_pil.copy()
        draw_ov = ImageDraw.Draw(overlay)
        draw_ov.rectangle([bx, by, bx + box_w, by + box_h], fill=(0, 0, 0))
        img_pil = Image.blend(img_pil, overlay, alpha=0.55)
        draw = ImageDraw.Draw(img_pil)

        r, g, b = 255, 255, 255
        for i, line in enumerate(text_lines):
            draw.text((bx + 8, by + 8 + i * line_h), line, fill=(r, g, b), font=font)

        return np.array(img_pil, dtype=np.uint8)

    @staticmethod
    def _write_report(
        report_path: Path,
        image_data: ImageData,
        scale: ScaleResult,
        capsid: CapsidResult,
        measure: MeasureResult,
    ) -> None:
        data = {
            "image": image_data.source_path.name,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "scale": {
                "px_per_nm": round(scale.px_per_nm, 6),
                "scale_text": scale.scale_text,
                "scale_nm": scale.scale_nm,
                "bar_bbox": list(scale.bar_bbox),
                "source": scale.source,
                "confidence": scale.confidence,
            },
            "capsid": {
                "center": list(capsid.center),
                "radius_px": capsid.radius,
                "radius_nm": round(capsid.radius / scale.px_per_nm, 2) if scale.px_per_nm > 0 else None,
            },
            "fibrils": {
                "count": len(measure.fibrils),
                "lengths_nm": {
                    "min": round(measure.min_nm, 2),
                    "mean": round(measure.mean_nm, 2),
                    "max": round(measure.max_nm, 2),
                    "std": round(measure.std_nm, 2),
                },
                "histogram": measure.histogram,
            },
        }
        report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
