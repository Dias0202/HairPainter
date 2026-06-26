"""Pipeline orchestrator — wires all services into a single processing chain."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hairpainter.services.capsid.capsid_service import CapsidService
from hairpainter.services.io.io_service import IOService
from hairpainter.services.measure.measure_service import MeasureService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.render.render_service import RenderService
from hairpainter.services.scale.scale_service import ScaleService
from hairpainter.services.segment.segment_service import SegmentService
from hairpainter.utils.types import PipelineInput, PipelineResult


@dataclass
class PipelineConfig:
    frangi_threshold: float = 0.05
    min_fibril_px: int = 15
    use_sam2: bool = False
    # Preprocessing: CLAHE clip=4.0 / tile=16 ("clahe_strong") is the winner of
    # the preprocessing experiment (scripts/experiment_preprocess.py) — marginally
    # better recall than the old 2.0/8 default.  Set 2.0/8 to restore v2.5.
    clahe_clip_limit: float = 4.0
    clahe_tile_grid: int = 16
    # Segmentation zone / capsid-exclusion geometry.  Defaults reproduce v2.5;
    # the experiment found no setting that beats them on the composite score
    # (the F1 ceiling is physical — SDD section 11), but they are exposed so the
    # operating point (recall vs precision vs capsid intrusion) can be tuned.
    #   capsid_mask_frac < 1.0  -> expose the inner peri-capsid ring (+recall, +capsid FP)
    #   extend_inward_to_frac>0 -> anchor roots to the surface (+recall, longer fibrils)
    #   zone_outer_frac         -> outer radial limit (smaller -> +precision, -recall)
    zone_inner_frac: float = 0.85
    zone_outer_frac: float = 2.0
    capsid_mask_frac: float = 1.0
    extend_inward_to_frac: float = 0.0
    # Optional deep-learning path: a trained U-Net beats the classical Frangi
    # ceiling (F1@5px ~0.41 vs ~0.31, SDD section 11.4).  Opt-in: needs PyTorch
    # and a checkpoint from scripts/train_unet.py.  Falls back to classical if
    # the checkpoint is missing or torch is unavailable.
    use_unet: bool = False
    unet_ckpt: str = ""
    unet_threshold: float = 0.45


# Signature for progress callback: (step_name: str, percent: int)
ProgressCallback = Callable[[str, int], None]

_STEPS = [
    ("Carregando imagem", 5),
    ("Pré-processando", 15),
    ("Detectando escala", 30),
    ("Detectando capsídeo", 45),
    ("Segmentando fibrilas", 65),
    ("Medindo fibrilas", 80),
    ("Gerando entregáveis", 95),
    ("Concluído", 100),
]


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        cfg = config or PipelineConfig()
        self._cfg = cfg
        self._io = IOService()
        self._pre = PreprocessService(
            clip_limit=cfg.clahe_clip_limit,
            tile_grid=cfg.clahe_tile_grid,
        )
        self._scale = ScaleService()
        self._capsid = CapsidService()
        self._segment = SegmentService(
            frangi_threshold=cfg.frangi_threshold,
            min_fibril_px=cfg.min_fibril_px,
            zone_inner_frac=cfg.zone_inner_frac,
            zone_outer_frac=cfg.zone_outer_frac,
            capsid_mask_frac=cfg.capsid_mask_frac,
            extend_inward_to_frac=cfg.extend_inward_to_frac,
        )
        self._measure = MeasureService()
        self._render = RenderService()

    def run(
        self,
        pipeline_input: PipelineInput,
        progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        result = PipelineResult(input_path=pipeline_input.image_path)

        def _progress(step: str, pct: int) -> None:
            if progress:
                progress(step, pct)

        try:
            _progress("Carregando imagem", 5)
            image_data = self._io.load(pipeline_input.image_path)

            _progress("Pré-processando", 15)
            enhanced = self._pre.enhance(image_data)

            _progress("Detectando escala", 30)
            scale = self._scale.detect(image_data)
            result.scale = scale

            _progress("Detectando capsídeo", 45)
            capsid = self._capsid.detect(enhanced)
            result.capsid = capsid

            _progress("Segmentando fibrilas", 65)
            segment = self._run_segment(enhanced, capsid)
            result.segment = segment

            _progress("Medindo fibrilas", 80)
            measure = self._measure.measure(segment, scale)
            result.measure = measure

            _progress("Gerando entregáveis", 95)
            stem = pipeline_input.image_path.stem
            img_output_dir = pipeline_input.output_dir / stem
            render = self._render.render(
                image_data=image_data,
                segment=segment,
                measure=measure,
                scale=scale,
                capsid=capsid,
                output_dir=img_output_dir,
            )
            result.render = render

            result.success = True
            _progress("Concluído", 100)

        except Exception as exc:
            result.success = False
            result.error = str(exc)
            _progress(f"Erro: {exc}", 0)

        return result

    def _run_segment(self, enhanced, capsid):
        """Classical segmentation, or the U-Net path when configured and usable."""
        cfg = self._cfg
        if cfg.use_unet and cfg.unet_ckpt and Path(cfg.unet_ckpt).exists():
            try:
                from hairpainter.services.segment.unet import segment_unet

                return segment_unet(
                    enhanced, capsid, cfg.unet_ckpt,
                    threshold=cfg.unet_threshold,
                    min_fibril_px=cfg.min_fibril_px,
                    zone_inner_frac=cfg.zone_inner_frac,
                    zone_outer_frac=cfg.zone_outer_frac,
                )
            except Exception as exc:  # noqa: BLE001 — fall back to classical
                print(f"[U-Net indisponível, usando clássico: {exc}]")
        return self._segment.segment(enhanced, capsid)

    def run_batch(
        self,
        inputs: list[PipelineInput],
        progress: ProgressCallback | None = None,
    ) -> list[PipelineResult]:
        results = []
        total = len(inputs)
        for i, inp in enumerate(inputs):
            def _scaled_progress(step: str, pct: int, i=i, total=total) -> None:
                if progress:
                    global_pct = int((i * 100 + pct) / total)
                    progress(f"[{i + 1}/{total}] {step}", global_pct)

            results.append(self.run(inp, _scaled_progress))
        return results
