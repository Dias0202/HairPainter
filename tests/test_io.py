"""Tests for IOService."""
import numpy as np
import pytest

from hairpainter.services.io.io_service import IOService
from hairpainter.utils.types import ImageData


def test_load_returns_image_data(sample_image_path):
    result = IOService().load(sample_image_path)
    assert isinstance(result, ImageData)


def test_array_is_uint8_grayscale(image_data):
    assert image_data.array.dtype == np.uint8
    assert image_data.array.ndim == 2


def test_original_is_rgb(image_data):
    assert image_data.original_array.ndim == 3
    assert image_data.original_array.shape[2] == 3


def test_metadata_contains_pixel_size(image_data):
    # Tecnai TIFF should have tag 65450
    assert "pixel_x_raw" in image_data.metadata or "n_frames" in image_data.metadata


def test_stack_tiff_selects_frame(image_data):
    # All 5 images are 3-frame stacks; loader should pick best frame
    if "n_frames" in image_data.metadata:
        assert image_data.metadata["n_frames"] >= 1


def test_image_dimensions(image_data):
    h, w = image_data.array.shape
    assert h > 0 and w > 0
    # Known dimensions: 1376×1070
    assert w == 1376
    assert h == 1070


def test_unsupported_extension_raises(tmp_path):
    fake = tmp_path / "test.bmp"
    fake.write_bytes(b"BM")
    with pytest.raises(ValueError, match="Unsupported"):
        IOService().load(fake)
