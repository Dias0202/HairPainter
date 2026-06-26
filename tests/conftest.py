"""Shared pytest fixtures."""
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "Data"
RAW_DIR = DATA_DIR / "Raw"
SVG_DIR = DATA_DIR / "Manual_paint"


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    """First available raw TIF image."""
    candidates = sorted(RAW_DIR.glob("*.tif"))
    if not candidates:
        pytest.skip("No TIF images found in Data/Raw/")
    return candidates[0]


@pytest.fixture(scope="session")
def all_image_paths() -> list[Path]:
    return sorted(RAW_DIR.glob("*.tif"))


@pytest.fixture(scope="session")
def image_data(sample_image_path):
    from hairpainter.services.io.io_service import IOService
    return IOService().load(sample_image_path)


@pytest.fixture(scope="session")
def enhanced(image_data):
    from hairpainter.services.preprocess.preprocess_service import PreprocessService
    return PreprocessService().enhance(image_data)
