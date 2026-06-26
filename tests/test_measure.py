"""Tests for MeasureService."""
import pytest

from hairpainter.services.capsid.capsid_service import CapsidService
from hairpainter.services.measure.measure_service import MeasureService
from hairpainter.services.scale.scale_service import ScaleService
from hairpainter.services.segment.segment_service import SegmentService
from hairpainter.utils.types import MeasureResult, ScaleResult


@pytest.fixture(scope="module")
def measure_result(image_data, enhanced):
    capsid = CapsidService().detect(enhanced)
    segment = SegmentService().segment(enhanced, capsid)
    scale = ScaleService().detect(image_data)

    # Override scale if not detected to allow test to run
    if scale.px_per_nm <= 0:
        from hairpainter.utils.types import ScaleResult
        scale = ScaleResult(
            px_per_nm=0.5,
            bar_bbox=(0, 0, 0, 0),
            scale_text="mock",
            scale_nm=2.0,
            source="manual",
        )

    return MeasureService().measure(segment, scale)


def test_measure_returns_result(measure_result):
    assert isinstance(measure_result, MeasureResult)


def test_lengths_are_positive(measure_result):
    for f in measure_result.fibrils:
        assert f.length_nm > 0


def test_min_le_mean_le_max(measure_result):
    if measure_result.fibrils:
        assert measure_result.min_nm <= measure_result.mean_nm <= measure_result.max_nm


def test_histogram_has_bins_and_counts(measure_result):
    if measure_result.fibrils:
        hist = measure_result.histogram
        assert "bins" in hist and "counts" in hist
        assert len(hist["counts"]) > 0


def test_zero_scale_raises(enhanced, image_data):
    capsid = CapsidService().detect(enhanced)
    segment = SegmentService().segment(enhanced, capsid)
    bad_scale = ScaleResult(
        px_per_nm=0.0,
        bar_bbox=(0, 0, 0, 0),
        scale_text="",
        scale_nm=0.0,
        source="manual",
    )
    with pytest.raises(ValueError, match="Scale not calibrated"):
        MeasureService().measure(segment, bad_scale)
