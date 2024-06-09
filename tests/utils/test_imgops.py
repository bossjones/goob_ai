from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest_asyncio
import torch

from goob_ai.utils.imgops import auto_split_upscale, bgr_to_rgb, bgra_to_rgba, convert_image_from_hwc_to_chw, handle_predict_one
from PIL import Image
import torch
from goob_ai.utils.imgops import convert_tensor_to_pil_image
import torch
from goob_ai.utils.imgops import convert_tensor_to_pil_image

import pytest
import pytest_asyncio
from PIL import Image
from pathlib import Path
from goob_ai.utils.imgops import get_all_corners_color
import asyncio
import torch
from goob_ai.utils.imgops import denorm, get_pil_image_channels, get_pixel_rgb, handle_autocrop, handle_autocrop_one, handle_get_dominant_color, handle_predict


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


def test_convert_tensor_to_pil_image(test_image):
    """Test convert_tensor_to_pil_image function."""
    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply convert_tensor_to_pil_image
    pil_image = convert_tensor_to_pil_image(test_image_tensor)

    # Check if the result is a PIL Image
    assert isinstance(pil_image, Image.Image)

    # Check if the dimensions match
    assert pil_image.size == (test_image.shape[1], test_image.shape[0])

    # Check if the mode is correct
    assert pil_image.mode == "RGB"

def test_convert_tensor_to_pil_image(test_image):
    """Test convert_tensor_to_pil_image function."""
    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply convert_tensor_to_pil_image
    pil_image = convert_tensor_to_pil_image(test_image_tensor)

    # Check if the result is a PIL Image
    assert isinstance(pil_image, Image.Image)

    # Check if the dimensions match
    assert pil_image.size == (test_image.shape[1], test_image.shape[0])

    # Check if the mode is correct
    assert pil_image.mode == "RGB"

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


@pytest.fixture
def test_image_path():
    return "tests/fixtures/screenshot_image_larger00013.PNG"

@pytest.fixture
def mock_model(mocker):
    model = mocker.Mock()
    model.name = "mock_model"
    return model

@pytest.mark.asyncio
async def test_handle_resize_one(mocker, test_image_path, mock_model):
    """Test handle_resize_one function."""
    from goob_ai.utils.imgops import handle_resize_one

    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(test_image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(test_image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=test_image_path)

    resized_image_path = handle_resize_one(
        images_filepath=test_image_path,
        model=mock_model,
        resize=True
    )

    assert resized_image_path == test_image_path

@pytest.mark.parametrize("input_tensor, expected_output", [
    (torch.tensor([0.0, 0.5, 1.0]), torch.tensor([-1.0, 0.0, 1.0])),
    (torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])),
    (torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0])),
    (np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0])),
    (np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
    (np.array([0.0, 1.0]), np.array([-1.0, 1.0])),
])
def test_norm(input_tensor, expected_output):
    """Test norm function with different input ranges."""
    from goob_ai.utils.imgops import norm

    output = norm(input_tensor)
    if isinstance(input_tensor, torch.Tensor):
        assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
    elif isinstance(input_tensor, np.ndarray):
        assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

@pytest.mark.asyncio
async def test_np2tensor(mocker):
    """Test np2tensor function."""
    from goob_ai.utils.imgops import np2tensor

    # Load the test image
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    test_image = np.array(Image.open(image_path))

    # Mock the np.ndarray type check
    mocker.patch("numpy.ndarray", return_value=True)

    # Test with default parameters
    tensor_image = await np2tensor(test_image)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # Test with bgr2rgb=False
    tensor_image = await np2tensor(test_image, bgr2rgb=False)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # Test with normalize=True
    tensor_image = await np2tensor(test_image, normalize=True)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.min() >= -1.0
    assert tensor_image.max() <= 1.0

    # Test with add_batch=False
    tensor_image = await np2tensor(test_image, add_batch=False)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.shape == (test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # Test with change_range=False
    tensor_image = await np2tensor(test_image, change_range=False)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.min() >= 0.0
    assert tensor_image.max() <= 255.0

    # Test with data_range=255.0
    tensor_image = await np2tensor(test_image, data_range=255.0)
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.min() >= 0.0
    assert tensor_image.max() <= 1.0
    """Test handle_resize_one function."""
    from goob_ai.utils.imgops import handle_resize_one

    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(test_image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(test_image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=test_image_path)

    resized_image_path = handle_resize_one(
        images_filepath=test_image_path,
        model=mock_model,
        resize=True
    )

    assert resized_image_path == test_image_path


@pytest.mark.asyncio
async def test_handle_resize_one_no_resize(mocker, test_image_path, mock_model):
    """Test handle_resize_one function without resizing."""
    from goob_ai.utils.imgops import handle_resize_one

    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(test_image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(test_image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=test_image_path)

    resized_image_path = handle_resize_one(
        images_filepath=test_image_path,
        model=mock_model,
        resize=False
    )

    assert resized_image_path == test_image_path

    """Test handle_predict_one function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
    mocker.patch("goob_ai.utils.imgops.predict_from_file", return_value=(Image.open(image_path), [(0, 0, 100, 100)]))

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    predict_result = await handle_predict_one(
        images_filepath=image_path,
        model=mock_model
    )

    assert isinstance(predict_result, tuple)
    assert isinstance(predict_result[0], Image.Image)
    assert isinstance(predict_result[1], list)
    assert len(predict_result[1]) == 1
    assert predict_result[1][0] == (0, 0, 100, 100)

@pytest.mark.asyncio
async def test_handle_predict(mocker):
    """Test handle_predict function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
    mocker.patch("goob_ai.utils.imgops.predict_from_file", return_value=(Image.open(image_path), [(0, 0, 100, 100)]))

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    predict_results = await handle_predict(
        images_filepaths=[image_path],
        model=mock_model
    )

    assert len(predict_results) == 1
    assert isinstance(predict_results[0], tuple)
    assert isinstance(predict_results[0][0], Image.Image)
    assert isinstance(predict_results[0][1], list)
    assert len(predict_results[0][1]) == 1
    assert predict_results[0][1][0] == (0, 0, 100, 100)
async def test_get_all_corners_color(mocker):
    """Test get_all_corners_color function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))

    urls = [image_path]
    corner_colors = get_all_corners_color(urls)

    assert corner_colors["top_left"] == (255, 255, 255)
    assert corner_colors["top_right"] == (255, 255, 255)
    assert corner_colors["bottom_left"] == (255, 255, 255)
    assert corner_colors["bottom_right"] == (255, 255, 255)


@pytest.mark.asyncio
async def test_get_pil_image_channels(mocker):
    """Test get_pil_image_channels function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))

    channels = get_pil_image_channels(image_path)

    assert channels == 3


@pytest.mark.asyncio
async def test_handle_autocrop(mocker):
    """Test handle_autocrop function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=image_path)

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    predict_results = [(Image.open(image_path), [(0, 0, 100, 100)])]

    cropped_image_paths = await handle_autocrop(
        images_filepaths=[image_path],
        model=mock_model,
        predict_results=predict_results
    )

    assert len(cropped_image_paths) == 1
    assert cropped_image_paths[0] == image_path


@pytest.mark.asyncio
async def test_handle_autocrop_one(mocker):
    """Test handle_autocrop_one function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=image_path)

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    predict_results = (Image.open(image_path), [(0, 0, 100, 100)])

    cropped_image_path = await handle_autocrop_one(
        images_filepath=image_path,
        model=mock_model,
        predict_results=predict_results
    )

    assert cropped_image_path == image_path


@pytest.mark.asyncio
async def test_handle_autocrop_one(mocker):
    """Test handle_autocrop_one function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
    mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=image_path)

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    predict_results = (Image.open(image_path), [(0, 0, 100, 100)])

    cropped_image_path = await handle_autocrop_one(
        images_filepath=image_path,
        model=mock_model,
        predict_results=predict_results
    )

    assert cropped_image_path == image_path


@pytest.mark.asyncio
async def test_handle_get_dominant_color_name(mocker):
    """Test handle_get_dominant_color function with return_type 'name'."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.get_all_corners_color", return_value={
        "top_left": (255, 255, 255),
        "top_right": (255, 255, 255),
        "bottom_left": (255, 255, 255),
        "bottom_right": (255, 255, 255),
    })
    mocker.patch("goob_ai.utils.imgops.convert_rgb_to_names", return_value="white")

    urls = [image_path]
    dominant_color = handle_get_dominant_color(urls, return_type="name")

    assert dominant_color == "white"

@pytest.mark.asyncio
async def test_handle_get_dominant_color_hex(mocker):
    """Test handle_get_dominant_color function with return_type 'hex'."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
    mocker.patch("goob_ai.utils.imgops.get_all_corners_color", return_value={
        "top_left": (255, 255, 255),
        "top_right": (255, 255, 255),
        "bottom_left": (255, 255, 255),
        "bottom_right": (255, 255, 255),
    })
    mocker.patch("goob_ai.utils.imgops.rgb2hex", return_value="#ffffff")

    urls = [image_path]
    dominant_color = handle_get_dominant_color(urls, return_type="hex")

    assert dominant_color == "#ffffff"

@pytest.mark.parametrize("input_tensor, min_max, expected_output", [
    (torch.tensor([-1.0, 0.0, 1.0]), (-1.0, 1.0), torch.tensor([0.0, 0.5, 1.0])),
    (torch.tensor([0.0, 0.5, 1.0]), (0.0, 1.0), torch.tensor([0.0, 0.5, 1.0])),
    (torch.tensor([0.0, 0.5, 1.0]), (0.0, 2.0), torch.tensor([0.0, 0.25, 0.5])),
])
def test_denorm(input_tensor, min_max, expected_output):
    """Test denorm function with different input ranges."""
    output = denorm(input_tensor, min_max)
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

@pytest.mark.asyncio
async def test_get_pixel_rgb(mocker):
    """Test get_pixel_rgb function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))

    image_pil = Image.open(image_path)
    color = get_pixel_rgb(image_pil)

    assert color == "white"

@pytest.mark.parametrize("input_array, min_max, expected_output", [
    (np.array([-1.0, 0.0, 1.0]), (-1.0, 1.0), np.array([0.0, 0.5, 1.0])),
    (np.array([0.0, 0.5, 1.0]), (0.0, 1.0), np.array([0.0, 0.5, 1.0])),
    (np.array([0.0, 0.5, 1.0]), (0.0, 2.0), np.array([0.0, 0.25, 0.5])),
])
def test_denorm_numpy(input_array, min_max, expected_output):
    """Test denorm function with numpy arrays and different input ranges."""
    output = denorm(input_array, min_max)
    assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
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


def test_convert_rgb_to_names(mocker):
    """Test convert_rgb_to_names function."""
    from goob_ai.utils.imgops import convert_rgb_to_names

    # Mock the CSS3_HEX_TO_NAMES and hex_to_rgb
    mock_css3_db = {
        "#ff0000": "red",
        "#00ff00": "green",
        "#0000ff": "blue",
    }
    mock_rgb_values = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    mocker.patch("goob_ai.utils.imgops.CSS3_HEX_TO_NAMES", mock_css3_db)
    mocker.patch("goob_ai.utils.imgops.hex_to_rgb", side_effect=lambda hex: mock_rgb_values[list(mock_css3_db.keys()).index(hex)])

    # Mock the KDTree query method
    mock_kdtree = mocker.patch("goob_ai.utils.imgops.KDTree.query", return_value=(0, 0))

    # Test with a sample RGB tuple
    rgb_tuple = (255, 0, 0)
    color_name = convert_rgb_to_names(rgb_tuple)

    # Check if the color name is correct
    assert color_name == "red"
    
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
