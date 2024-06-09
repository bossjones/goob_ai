import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from src.goob_ai.utils.imgops import auto_split_upscale

@pytest.fixture
def test_image():
    image_path = Path("tests/fixtures/screenshot_image_larger00013.PNG")
    return np.array(Image.open(image_path))

def dummy_upscale_function(image: np.ndarray) -> np.ndarray:
    """A dummy upscale function that just returns the input image."""
    return image

def test_auto_split_upscale_no_split(test_image):
    """Test auto_split_upscale without splitting the image."""
    scale = 2
    overlap = 0
    upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (test_image.shape[0] * scale, test_image.shape[1] * scale, test_image.shape[2])
    assert depth == 1

def test_auto_split_upscale_with_split(test_image, mocker):
    """Test auto_split_upscale with splitting the image."""
    scale = 2
    overlap = 32

    # Mock the dummy upscale function to raise an error to force splitting
    mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (test_image.shape[0] * scale, test_image.shape[1] * scale, test_image.shape[2])
    assert depth > 1

def test_auto_split_upscale_max_depth(test_image, mocker):
    """Test auto_split_upscale with a maximum recursion depth."""
    scale = 2
    overlap = 32
    max_depth = 2

    # Mock the dummy upscale function to raise an error to force splitting
    mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    with pytest.raises(RecursionError):
        auto_split_upscale(test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)
