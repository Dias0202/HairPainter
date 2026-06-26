"""Tests for ScaleService."""
import numpy as np
import pytest

from hairpainter.services.scale.scale_service import ScaleService
from hairpainter.utils.types import ScaleResult


def test_detect_returns_scale_result(image_data):
    result = ScaleService().detect(image_data)
    assert isinstance(result, ScaleResult)


def test_scale_source_is_valid(image_data):
    result = ScaleService().detect(image_data)
    assert result.source in ("visual", "metadata", "manual")


def test_px_per_nm_positive_or_zero(image_data):
    result = ScaleService().detect(image_data)
    assert result.px_per_nm >= 0.0


def test_dark_bar_present_in_bottom_15_percent(image_data):
    """The scale bar background (dark region) should be in the bottom of the image."""
    gray = image_data.array
    h = gray.shape[0]
    bottom = gray[int(h * 0.85):, :]
    dark_fraction = (bottom < 40).sum() / bottom.size
    assert dark_fraction > 0.1, "Expected significant dark area in image bottom (scale bar)"


def test_scale_text_not_empty_if_visual(image_data):
    result = ScaleService().detect(image_data)
    if result.source == "visual":
        assert len(result.scale_text) > 0
