"""Tests for CapsidService and SegmentService."""
import numpy as np
import pytest

from hairpainter.services.capsid.capsid_service import CapsidService
from hairpainter.services.segment.segment_service import SegmentService
from hairpainter.utils.types import CapsidResult, SegmentResult


# ---------------------------------------------------------------------------
# Synthetic fixtures — run without real TIF images
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_capsid() -> CapsidResult:
    """400x400 image with capsid centred at (200,200), radius 80."""
    h, w, cx, cy, r = 400, 400, 200, 200, 80
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 1, thickness=-1)
    return CapsidResult(center=(cx, cy), radius=r, mask=mask.astype(bool))


@pytest.fixture
def synthetic_image(synthetic_capsid) -> np.ndarray:
    """
    Synthetic grayscale image: bright background (200) with a dark capsid (60)
    and eight radial dark fibril stripes extending from the capsid surface.
    """
    h, w = 400, 400
    cx, cy = synthetic_capsid.center
    r = synthetic_capsid.radius

    img = np.full((h, w), 200, dtype=np.uint8)

    # Dark capsid interior
    y_g, x_g = np.ogrid[:h, :w]
    dist = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2)
    img[dist < r] = 60

    # 8 radial dark fibrils (lines from r to 1.8*r)
    for angle_deg in np.arange(0, 360, 45):
        theta = np.radians(angle_deg)
        for rho in np.linspace(r, int(r * 1.8), 60):
            px = int(round(cx + rho * np.cos(theta)))
            py = int(round(cy + rho * np.sin(theta)))
            if 0 <= px < w and 0 <= py < h:
                img[py, px] = 80
            # ±1 pixel wide
            for dpx, dpy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                xi, yi = px + dpx, py + dpy
                if 0 <= xi < w and 0 <= yi < h:
                    img[yi, xi] = 90

    return img


# ---------------------------------------------------------------------------
# Tests using real TIF images (skipped if unavailable)
# ---------------------------------------------------------------------------

def test_capsid_detect_returns_result(enhanced):
    result = CapsidService().detect(enhanced)
    assert isinstance(result, CapsidResult)


def test_capsid_mask_shape_matches_image(image_data, enhanced):
    result = CapsidService().detect(enhanced)
    assert result.mask.shape == enhanced.shape


def test_capsid_center_within_image(enhanced):
    result = CapsidService().detect(enhanced)
    h, w = enhanced.shape
    cx, cy = result.center
    assert 0 <= cx < w
    assert 0 <= cy < h


def test_capsid_radius_positive(enhanced):
    result = CapsidService().detect(enhanced)
    assert result.radius > 0


def test_segment_returns_result(enhanced):
    capsid = CapsidService().detect(enhanced)
    result = SegmentService().segment(enhanced, capsid)
    assert isinstance(result, SegmentResult)


def test_segment_detects_fibrils(enhanced):
    capsid = CapsidService().detect(enhanced)
    result = SegmentService().segment(enhanced, capsid)
    assert result.n_fibrils > 0, "Expected fibrils to be detected"


def test_label_map_shape(enhanced):
    capsid = CapsidService().detect(enhanced)
    result = SegmentService().segment(enhanced, capsid)
    assert result.label_map.shape == enhanced.shape


def test_fibril_instances_have_valid_skeletons(enhanced):
    capsid = CapsidService().detect(enhanced)
    result = SegmentService(min_fibril_px=5).segment(enhanced, capsid)
    for f in result.fibrils[:10]:
        assert f.skeleton.dtype == bool
        assert f.length_px > 0


# ---------------------------------------------------------------------------
# Tests using synthetic image (always run, no real data required)
# ---------------------------------------------------------------------------

def test_segment_synthetic_returns_result(synthetic_image, synthetic_capsid):
    result = SegmentService(min_fibril_px=5).segment(synthetic_image, synthetic_capsid)
    assert isinstance(result, SegmentResult)
    assert result.label_map.shape == synthetic_image.shape


def test_anchoring_constraint_no_floating_fibrils(synthetic_image, synthetic_capsid):
    """All detected fibrils must touch or be close to the capsid surface."""
    r = synthetic_capsid.radius
    cx, cy = synthetic_capsid.center
    h, w = synthetic_image.shape
    y_g, x_g = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_g - cx) ** 2 + (y_g - cy) ** 2).astype(np.float32)
    dist_from_surface = np.abs(dist_from_center - r)

    svc = SegmentService(min_fibril_px=5, anchor_band_px=25)
    result = svc.segment(synthetic_image, synthetic_capsid)

    for fibril in result.fibrils:
        min_dist = float(dist_from_surface[fibril.skeleton].min())
        assert min_dist <= 25, (
            f"Fibril {fibril.id} is floating: min dist from surface = {min_dist:.1f}px"
        )


def test_background_subtract_highlights_dark_regions():
    """Background subtraction must return positive values where image < background."""
    svc = SegmentService()
    gray = np.full((100, 100), 150, dtype=np.uint8)
    gray[40:60, 40:60] = 80  # dark rectangle
    result = svc._background_subtract(gray, sigma=15)
    assert result.dtype == np.uint8
    # Dark region should have higher response than bright background
    assert result[50, 50] > result[10, 10]


def test_polar_frangi_output_shape():
    """Polar Frangi must return array with the same shape as input image."""
    svc = SegmentService()
    gray = np.random.randint(100, 200, (400, 400), dtype=np.uint8)
    cx, cy, r = 200, 200, 80
    max_r = int(r * 2.2)
    out = svc._polar_frangi(gray, cx, cy, r, max_r)
    assert out.shape == gray.shape
    assert out.dtype == np.float32
