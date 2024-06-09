from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest_asyncio
import torch

from goob_ai.utils.imgops import auto_split_upscale, bgr_to_rgb
from PIL import Image

import pytest


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
    assert upscaled_image.shape == (test_image.shape[0], test_image.shape[1], test_image.shape[2])
    assert depth == 1


def test_auto_split_upscale_with_split(test_image, pytest_mocker):
    """Test auto_split_upscale with splitting the image."""
    scale = 2
    overlap = 32

    # Mock the dummy upscale function to raise an error to force splitting
    pytest_mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (test_image.shape[0] * scale, test_image.shape[1] * scale, test_image.shape[2])
    assert depth > 1


def test_auto_split_upscale_max_depth(test_image, pytest_mocker):
    """Test auto_split_upscale with a maximum recursion depth."""
    scale = 2
    overlap = 32
    max_depth = 2

    # Mock the dummy upscale function to raise an error to force splitting
    pytest_mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    with pytest.raises(RecursionError):
        auto_split_upscale(test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)


def test_bgr_to_rgb(test_image):
    """Test bgr_to_rgb function."""
    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply bgr_to_rgb
    rgb_image_tensor = bgr_to_rgb(test_image_tensor)

    # Convert back to numpy for comparison
    rgb_image = rgb_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(rgb_image[:, :, 0], test_image[:, :, 2])  # R channel
    assert np.array_equal(rgb_image[:, :, 1], test_image[:, :, 1])  # G channel
    assert np.array_equal(rgb_image[:, :, 2], test_image[:, :, 0])  # B channel


@pytest.mark.asyncio
async def test_bgr_to_rgb_async(async_test_image):
    """Test bgr_to_rgb function (async)."""
    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(async_test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply bgr_to_rgb
    rgb_image_tensor = bgr_to_rgb(test_image_tensor)

    # Convert back to numpy for comparison
    rgb_image = rgb_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(rgb_image[:, :, 0], async_test_image[:, :, 2])  # R channel
    assert np.array_equal(rgb_image[:, :, 1], async_test_image[:, :, 1])  # G channel
    assert np.array_equal(rgb_image[:, :, 2], async_test_image[:, :, 0])  # B channel

    """Test auto_split_upscale with splitting the image (async)."""
    scale = 2
    overlap = 32

    # Mock the dummy upscale function to raise an error to force splitting
    pytest_mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    upscaled_image, depth = auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (async_test_image.shape[0] * scale, async_test_image.shape[1] * scale, async_test_image.shape[2])
    assert depth > 1


@pytest.mark.asyncio
async def test_auto_split_upscale_max_depth_async(async_test_image, pytest_mocker):
    """Test auto_split_upscale with a maximum recursion depth (async)."""
    scale = 2
    overlap = 32
    max_depth = 2

    # Mock the dummy upscale function to raise an error to force splitting
    pytest_mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    with pytest.raises(RecursionError):
        auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)


@pytest_asyncio.fixture
async def async_test_image():
    image_path = Path("tests/fixtures/screenshot_image_larger00013.PNG")
    return np.array(Image.open(image_path))


@pytest.mark.asyncio
async def test_auto_split_upscale_no_split_async(async_test_image):
    """Test auto_split_upscale without splitting the image (async)."""
    scale = 2
    overlap = 0
    upscaled_image, depth = auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (async_test_image.shape[0], async_test_image.shape[1], async_test_image.shape[2])
    assert depth == 1


# @pytest.mark.asyncio
# async def test_auto_split_upscale_with_split_async(async_test_image, mocker):
#     """Test auto_split_upscale with splitting the image (async)."""
#     scale = 2
#     overlap = 32

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("tests.utils.test_imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     upscaled_image, depth = auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap)
#     assert upscaled_image.shape == (async_test_image.shape[0] * scale, async_test_image.shape[1] * scale, async_test_image.shape[2])
#     assert depth > 1


# @pytest.mark.asyncio
# async def test_auto_split_upscale_max_depth_async(async_test_image, mocker):
#     """Test auto_split_upscale with a maximum recursion depth (async)."""
#     scale = 2
#     overlap = 32
#     max_depth = 2

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("tests.utils.test_imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     with pytest.raises(RecursionError):
#         auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)
