from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest_asyncio
import torch

from goob_ai.utils.imgops import auto_split_upscale, bgr_to_rgb, bgra_to_rgba, convert_image_from_hwc_to_chw
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


def test_bgra_to_rgba(test_image):
    """Test bgra_to_rgba function."""
    # Convert the test image to a tensor with an alpha channel
    test_image_tensor = torch.from_numpy(np.dstack((test_image, np.full(test_image.shape[:2], 255)))).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply bgra_to_rgba
    rgba_image_tensor = bgra_to_rgba(test_image_tensor)

    # Convert back to numpy for comparison
    rgba_image = rgba_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(rgba_image[:, :, 0], test_image[:, :, 2])  # R channel
    assert np.array_equal(rgba_image[:, :, 1], test_image[:, :, 1])  # G channel
    assert np.array_equal(rgba_image[:, :, 2], test_image[:, :, 0])  # B channel
    assert np.array_equal(rgba_image[:, :, 3], np.full(test_image.shape[:2], 255))  # A channel

def test_convert_image_from_hwc_to_chw(test_image):
    """Test convert_image_from_hwc_to_chw function."""
    # Convert the test image to HWC format
    test_image_hwc = test_image

    # Apply convert_image_from_hwc_to_chw
    chw_image_tensor = convert_image_from_hwc_to_chw(test_image_hwc)

    # Check if the shape is correctly converted to CHW
    assert chw_image_tensor.shape == (test_image_hwc.shape[2], test_image_hwc.shape[0], test_image_hwc.shape[1])
def test_convert_pil_image_to_rgb_channels(test_image, mocker):
    """Test convert_pil_image_to_rgb_channels function."""
    from goob_ai.utils.imgops import convert_pil_image_to_rgb_channels

    # Mock the get_pil_image_channels function to return 4 channels
    mocker.patch("goob_ai.utils.imgops.get_pil_image_channels", return_value=4)

    # Apply convert_pil_image_to_rgb_channels
    converted_image = convert_pil_image_to_rgb_channels("tests/fixtures/screenshot_image_larger00013.PNG")

    # Check if the image is converted to RGB
    assert converted_image.mode == "RGB"

def test_convert_pil_image_to_torch_tensor(test_image):
    """Test convert_pil_image_to_torch_tensor function."""
    from goob_ai.utils.imgops import convert_pil_image_to_torch_tensor

    # Convert the test image to a PIL image
    test_image_pil = Image.fromarray(test_image)

    # Apply convert_pil_image_to_torch_tensor
    tensor_image = convert_pil_image_to_torch_tensor(test_image_pil)

    # Check if the tensor shape is correct (C, H, W)
    assert tensor_image.shape == (test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # Check if the tensor values are in the correct range [0, 1]
    assert tensor_image.min() >= 0.0
    assert tensor_image.max() <= 1.0


@pytest.mark.asyncio
async def test_convert_pil_image_to_torch_tensor_async(async_test_image):
    """Test convert_pil_image_to_torch_tensor function (async)."""
    from goob_ai.utils.imgops import convert_pil_image_to_torch_tensor

    # Convert the test image to a PIL image
    async_test_image_pil = Image.fromarray(async_test_image)

    # Apply convert_pil_image_to_torch_tensor
    tensor_image = convert_pil_image_to_torch_tensor(async_test_image_pil)

    # Check if the tensor shape is correct (C, H, W)
    assert tensor_image.shape == (async_test_image.shape[2], async_test_image.shape[0], async_test_image.shape[1])

    # Check if the tensor values are in the correct range [0, 1]
    assert tensor_image.min() >= 0.0
    assert tensor_image.max() <= 1.0


@pytest.mark.asyncio
async def test_convert_pil_image_to_rgb_channels_async(async_test_image, mocker):
    """Test convert_pil_image_to_rgb_channels function (async)."""
    from goob_ai.utils.imgops import convert_pil_image_to_rgb_channels

    # Mock the get_pil_image_channels function to return 4 channels
    mocker.patch("goob_ai.utils.imgops.get_pil_image_channels", return_value=4)

    # Apply convert_pil_image_to_rgb_channels
    converted_image = convert_pil_image_to_rgb_channels("tests/fixtures/screenshot_image_larger00013.PNG")

    # Check if the image is converted to RGB
    assert converted_image.mode == "RGB"
    """Test bgra_to_rgba function (async)."""
    # Convert the test image to a tensor with an alpha channel
    test_image_tensor = torch.from_numpy(np.dstack((async_test_image, np.full(async_test_image.shape[:2], 255)))).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply bgra_to_rgba
    rgba_image_tensor = bgra_to_rgba(test_image_tensor)

    # Convert back to numpy for comparison
    rgba_image = rgba_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(rgba_image[:, :, 0], async_test_image[:, :, 2])  # R channel
    assert np.array_equal(rgba_image[:, :, 1], async_test_image[:, :, 1])  # G channel
    assert np.array_equal(rgba_image[:, :, 2], async_test_image[:, :, 0])  # B channel
    assert np.array_equal(rgba_image[:, :, 3], np.full(async_test_image.shape[:2], 255))  # A channel

@pytest.mark.asyncio
async def test_bgr_to_rgb_async(async_test_image, mocker):
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
    mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

    upscaled_image, depth = auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (async_test_image.shape[0] * scale, async_test_image.shape[1] * scale, async_test_image.shape[2])
    assert depth > 1


@pytest.mark.asyncio
async def test_auto_split_upscale_max_depth_async(async_test_image, mocker):
    """Test auto_split_upscale with a maximum recursion depth (async)."""
    scale = 2
    overlap = 32
    max_depth = 2

    # Mock the dummy upscale function to raise an error to force splitting
    mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

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
