from __future__ import annotations

import asyncio
import os

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest_asyncio
import torch

from goob_ai.utils.devices import get_device
from goob_ai.utils.imgops import (
    auto_split_upscale,
    bgr_to_rgb,
    bgra_to_rgba,
    convert_image_from_hwc_to_chw,
    convert_tensor_to_pil_image,
    denorm,
    get_all_corners_color,
    get_pil_image_channels,
    get_pixel_rgb,
    handle_autocrop,
    handle_autocrop_one,
    handle_get_dominant_color,
    handle_predict,
    handle_predict_one,
    predict_from_file,
    resize_and_pillarbox,
    resize_image_and_bbox,
    rgb_to_bgr,
    setup_model,
)
from PIL import Image

import pytest
import pytest_mock


@pytest.fixture
def test_image():
    image_path = Path("tests/fixtures/screenshot_image_larger00013.PNG")
    return np.array(Image.open(image_path))


def dummy_upscale_function(image: np.ndarray) -> np.ndarray:
    """A dummy upscale function that just returns the input image."""
    return image


@pytest.mark.imgops
def test_auto_split_upscale_no_split(test_image):
    """Test auto_split_upscale without splitting the image."""
    scale = 2
    overlap = 0
    upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (test_image.shape[0], test_image.shape[1], test_image.shape[2])
    assert depth == 1


# def test_auto_split_upscale_with_split(test_image, mocker):
#     """Test auto_split_upscale with splitting the image."""
#     scale = 2
#     overlap = 32

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
#     assert upscaled_image.shape == (test_image.shape[0] * scale, test_image.shape[1] * scale, test_image.shape[2])
#     assert depth > 1


# def test_auto_split_upscale_max_depth(test_image, mocker):
#     """Test auto_split_upscale with a maximum recursion depth."""
#     scale = 2
#     overlap = 32
#     max_depth = 2

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     with pytest.raises(RecursionError):
#         auto_split_upscale(test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)


@pytest.mark.imgops
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


@pytest.mark.imgops
def test_convert_tensor_to_pil_image1(test_image):
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


@pytest.mark.imgops
def test_convert_tensor_to_pil_image2(test_image):
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


@pytest.mark.imgops
def test_convert_pil_image_to_rgb_channels(test_image, mocker):
    """Test convert_pil_image_to_rgb_channels function."""
    from goob_ai.utils.imgops import convert_pil_image_to_rgb_channels

    # Apply convert_pil_image_to_rgb_channels
    converted_image = convert_pil_image_to_rgb_channels("tests/fixtures/screenshot_image_larger00013.PNG")

    # Check if the image is converted to RGB
    assert converted_image.mode == "RGB"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.imgops
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


# @pytest.mark.imgops
# @pytest.mark.parametrize(
#     "background_color, expected_color",
#     [
#         ("white", (255, 255, 255)),
#         ("darkmode", (22, 32, 42)),
#     ],
# )
# def test_resize_and_pillarbox(test_image_path, background_color, expected_color):
#     """Test resize_and_pillarbox function."""
#     from goob_ai.utils.imgops import get_pixel_rgb
#     from PIL import Image

#     # Load the test image
#     test_image = Image.open(test_image_path)

#     # Apply resize_and_pillarbox
#     resized_image = resize_and_pillarbox(test_image, 1080, 1350, background=background_color)

#     # Check if the result is a PIL Image
#     assert isinstance(resized_image, Image.Image)

#     # Check if the dimensions match
#     assert resized_image.size == (1080, 1350)

#     # Check if the background color is correct
#     assert resized_image.getpixel((0, 0)) == expected_color


@pytest.mark.imgops
@pytest.mark.asyncio
async def test_setup_model(mocker):
    """Test setup_model function."""
    from goob_ai.utils.imgops import load_model, setup_model

    # Mock the load_model function
    mock_load_model = mocker.patch("goob_ai.utils.imgops.load_model", return_value=mocker.Mock())

    # Call the setup_model function
    model = setup_model()

    # Assertions
    mock_load_model.assert_called_once_with(mocker.ANY, model_name="ScreenNetV1.pth")
    assert isinstance(model, mocker.Mock)
    """Test predict_from_file function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    # Mock the necessary functions and objects
    mock_image = Image.open(image_path)
    mock_bboxes = [(0, 0, 100, 100)]

    mocker.patch("goob_ai.utils.imgops.convert_pil_image_to_rgb_channels", return_value=mock_image)
    mocker.patch("goob_ai.utils.imgops.pred_and_store", return_value=mock_bboxes)

    mock_model = mocker.Mock()
    mock_model.name = "mock_model"

    # Call the function
    result_image, result_bboxes = predict_from_file(image_path, mock_model)

    # Assertions
    assert isinstance(result_image, Image.Image)
    assert result_image == mock_image
    assert result_bboxes == mock_bboxes


@pytest.mark.imgops
@pytest.mark.integration
async def test_pred_and_store(mocker):
    """Test pred_and_store function."""
    from goob_ai.utils.imgops import pred_and_store, setup_model

    # from goob_ai.utils.imgops import load_model, setup_model
    model = setup_model()

    # Mock the model to return dummy bounding boxes
    # mock_model = mocker.Mock()
    # mock_model.return_value = torch.tensor([[0, 0, 100, 100]])

    # Call the pred_and_store function
    paths = [Path("tests/fixtures/screenshot_image_larger00013.PNG")]
    result = pred_and_store(paths, model)

    # Check if the result is a list of dictionaries
    assert isinstance(result, torch.Tensor)
    assert len(result) == 1
    # assert isinstance(result[0], dict)
    # assert "image_path" in result[0]
    # assert "bounding_boxes" in result[0]


@pytest.mark.imgops
async def test_np2tensor():
    """Test np2tensor function."""
    from goob_ai.utils.imgops import np2tensor

    # Load the test image
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    test_image = np.array(Image.open(image_path))

    # Test with default parameters
    tensor_image = await np2tensor(test_image)
    assert isinstance(tensor_image, torch.Tensor)
    expected = torch.Size([1, 3, 2556, 1179])
    assert tensor_image.shape == expected

    assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # # Test with bgr2rgb=False
    # tensor_image = await np2tensor(test_image, bgr2rgb=False)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # # Test with normalize=True
    # tensor_image = await np2tensor(test_image, normalize=True)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.min() >= -1.0
    # assert tensor_image.max() <= 1.0

    # # Test with add_batch=False
    # tensor_image = await np2tensor(test_image, add_batch=False)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.shape == (test_image.shape[2], test_image.shape[0], test_image.shape[1])

    # # Test with change_range=False
    # tensor_image = await np2tensor(test_image, change_range=False)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.min() >= 0.0
    # assert tensor_image.max() <= 255.0

    # # Test with data_range=255.0
    # tensor_image = await np2tensor(test_image, data_range=255.0)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.min() >= 0.0
    # assert tensor_image.max() <= 1.0
    # """Test handle_resize_one function."""
    # from goob_ai.utils.imgops import handle_resize_one

    # resized_image_path = handle_resize_one(images_filepath=test_image_path, model=mock_model, resize=True)

    # assert resized_image_path == test_image_path


@pytest.mark.imgops
@pytest.mark.parametrize(
    "input_tensor, expected_output",
    [
        (torch.tensor([0.0, 0.5, 1.0]), torch.tensor([-1.0, 0.0, 1.0])),
        (torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 1.0])),
        (np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0])),
        (np.array([0.0, 0.25, 0.5, 0.75, 1.0]), np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
        (np.array([0.0, 1.0]), np.array([-1.0, 1.0])),
    ],
)
def test_norm(input_tensor, expected_output):
    """Test norm function with different input ranges."""
    from goob_ai.utils.imgops import norm

    output = norm(input_tensor)
    # sourcery skip: no-conditionals-in-tests
    if isinstance(input_tensor, torch.Tensor):
        assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
    elif isinstance(input_tensor, np.ndarray):
        assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


# @pytest.mark.asyncio
# async def test_np2tensor(mocker):
#     """Test np2tensor function."""
#     from goob_ai.utils.imgops import np2tensor

#     # Load the test image
#     image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
#     test_image = np.array(Image.open(image_path))

#     # Mock the np.ndarray type check
#     mocker.patch("numpy.ndarray", return_value=True)

#     # Test with default parameters
#     tensor_image = await np2tensor(test_image)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

#     # Test with bgr2rgb=False
#     tensor_image = await np2tensor(test_image, bgr2rgb=False)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])

#     # Test with normalize=True
#     tensor_image = await np2tensor(test_image, normalize=True)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.min() >= -1.0
#     assert tensor_image.max() <= 1.0

#     # Test with add_batch=False
#     tensor_image = await np2tensor(test_image, add_batch=False)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.shape == (test_image.shape[2], test_image.shape[0], test_image.shape[1])

#     # Test with change_range=False
#     tensor_image = await np2tensor(test_image, change_range=False)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.min() >= 0.0
#     assert tensor_image.max() <= 255.0

#     # Test with data_range=255.0
#     tensor_image = await np2tensor(test_image, data_range=255.0)
#     assert isinstance(tensor_image, torch.Tensor)
#     assert tensor_image.min() >= 0.0
#     assert tensor_image.max() <= 1.0


@pytest.mark.imgops
@pytest.mark.asyncio
async def test_tensor2np(mocker):
    """Test tensor2np function."""
    from goob_ai.utils.imgops import tensor2np

    # Load the test image
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    test_image = np.array(Image.open(image_path))

    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Test with default parameters
    numpy_image = await tensor2np(test_image_tensor)
    assert isinstance(numpy_image, np.ndarray)
    assert numpy_image.shape == (test_image.shape[0], test_image.shape[1], test_image.shape[2])

    # Test with rgb2bgr=False
    numpy_image = await tensor2np(test_image_tensor, rgb2bgr=False)
    assert isinstance(numpy_image, np.ndarray)
    assert numpy_image.shape == (test_image.shape[0], test_image.shape[1], test_image.shape[2])

    # Test with remove_batch=False
    test_image_tensor = test_image_tensor.unsqueeze(0)  # Add batch dimension
    numpy_image = await tensor2np(test_image_tensor, remove_batch=False)
    assert isinstance(numpy_image, np.ndarray)
    # assert numpy_image.shape == (test_image_tensor.shape[1], test_image_tensor.shape[2], test_image_tensor.shape[3])

    # # Test with denormalize=True
    # test_image_tensor = (test_image_tensor - 0.5) * 2.0  # Normalize to [-1, 1]
    # numpy_image = await tensor2np(test_image_tensor, denormalize=True)
    # assert isinstance(numpy_image, np.ndarray)
    # assert numpy_image.min() >= 0.0
    # assert numpy_image.max() <= 255.0

    # # Test with change_range=False
    # numpy_image = await tensor2np(test_image_tensor, change_range=False)
    # assert isinstance(numpy_image, np.ndarray)
    # assert numpy_image.min() >= -1.0
    # assert numpy_image.max() <= 1.0

    # # Test with data_range=1.0
    # numpy_image = await tensor2np(test_image_tensor, data_range=1.0)
    # assert isinstance(numpy_image, np.ndarray)
    # assert numpy_image.min() >= 0.0
    # assert numpy_image.max() <= 1.0
    # """Test handle_resize_one function."""
    # from goob_ai.utils.imgops import handle_resize_one

    # resized_image_path = handle_resize_one(images_filepath=test_image_path, model=mock_model, resize=True)

    # assert resized_image_path == test_image_path


@pytest.mark.imgops
def test_read_image_to_bgr(mocker):
    """Test read_image_to_bgr function."""
    from goob_ai.utils.imgops import read_image_to_bgr

    # Mock cv2.imread and cv2.cvtColor
    mock_image = np.array(Image.open("tests/fixtures/screenshot_image_larger00013.PNG"))
    # mocker.patch("cv2.imread", return_value=mock_image)
    # mocker.patch("cv2.cvtColor", return_value=mock_image)

    # Call the function
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    image, channels, height, width = read_image_to_bgr(image_path)

    # Assertions
    assert isinstance(image, np.ndarray)
    assert channels == 3
    assert height == mock_image.shape[0]
    assert width == mock_image.shape[1]


@pytest.mark.imgops
@pytest.mark.parametrize("return_percent_coords", [True, False])
def test_resize_image_and_bbox(mocker, test_image, return_percent_coords):
    """Test resize_image_and_bbox function."""
    from goob_ai.utils.imgops import resize_image_and_bbox

    # Mock the device
    test_device = get_device()

    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Create dummy bounding boxes
    boxes = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32).to(test_device)

    # Define target dimensions
    target_dims = (300, 300)

    # Call the function
    resized_image, resized_boxes = resize_image_and_bbox(
        test_image_tensor, boxes, dims=target_dims, return_percent_coords=return_percent_coords, device=test_device
    )

    # Check if the resized image has the correct dimensions
    assert resized_image.shape[1] == target_dims[0]
    assert resized_image.shape[2] == target_dims[1]

    # Check if the resized boxes have the correct dimensions
    # sourcery skip: no-conditionals-in-tests
    if return_percent_coords:
        assert torch.all(resized_boxes <= 1.0)
    else:
        assert torch.all(resized_boxes[:, 2] <= target_dims[0])
        assert torch.all(resized_boxes[:, 3] <= target_dims[1])
    # """Test handle_resize_one function without resizing."""
    # from goob_ai.utils.imgops import handle_resize_one

    # mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(test_image_path))
    # mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(test_image_path)))
    # mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
    # mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=test_image_path)

    # resized_image_path = handle_resize_one(images_filepath=test_image_path, model=mock_model, resize=False)

    # assert resized_image_path == test_image_path


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_handle_predict_one(mocker):
    """Test handle_predict_one function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    model = setup_model()
    # mock_model = mocker.Mock()
    # mock_model.name = "mock_model"

    test_image, test_bboxes = handle_predict_one(images_filepath=image_path, model=model)

    # assert isinstance(predict_result, tuple)
    assert isinstance(test_image, Image.Image)
    assert isinstance(test_bboxes, torch.Tensor)
    # TODO: test values
    # assert len(predict_result[1]) == 1
    # assert predict_result[1][0] == (0, 0, 100, 100)


@pytest.mark.imgops
def test_handle_predict(mocker):
    """Test handle_predict function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    model = setup_model()
    # mock_model = mocker.Mock()
    # mock_model.name = "mock_model"

    predict_results = handle_predict(images_filepaths=[image_path], model=model)
    assert isinstance(predict_results, list)
    # assert isinstance(test_bboxes, torch.Tensor)

    # assert len(predict_results) == 1
    # assert isinstance(predict_results[0], tuple)
    # assert isinstance(predict_results[0][0], Image.Image)
    # assert isinstance(predict_results[0][1], list)
    # assert len(predict_results[0][1]) == 1
    # assert predict_results[0][1][0] == (0, 0, 100, 100)


@pytest.mark.imgops
def test_get_all_corners_color():
    """Test get_all_corners_color function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    urls = [image_path]
    corner_colors = get_all_corners_color(urls)

    print(corner_colors)

    assert corner_colors["top_left"] == (23, 31, 41)
    assert corner_colors["top_right"] == (22, 31, 42)
    assert corner_colors["bottom_left"] == (23, 31, 42)
    assert corner_colors["bottom_right"] == (23, 31, 42)


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_get_pil_image_channels(mocker):
    """Test get_pil_image_channels function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    # 'pil_img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1179x2556 at 0x32AA189D0>,
    channels = get_pil_image_channels(image_path)

    assert channels == 3


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_handle_autocrop(mocker):
    """Test handle_autocrop function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    model = setup_model()
    # mock_model = mocker.Mock()
    # mock_model.name = "mock_model"

    predict_results = [(Image.open(image_path), [(0, 0, 100, 100)])]

    cropped_image_paths = handle_autocrop(images_filepaths=[image_path], model=model, predict_results=predict_results)

    assert len(cropped_image_paths) == 1
    assert "cropped-ObjLocModelV1-screenshot_image_larger00013.PNG" in cropped_image_paths[0]
    os.remove(cropped_image_paths[0])


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_handle_autocrop_one(mocker):
    """Test handle_autocrop_one function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    model = setup_model()
    # mock_model = mocker.Mock()
    # mock_model.name = "mock_model"

    predict_results = (Image.open(image_path), [(0, 0, 100, 100)])

    cropped_image_path = handle_autocrop_one(images_filepath=image_path, model=model, predict_results=predict_results)

    # assert cropped_image_path == image_path
    assert "cropped-ObjLocModelV1-screenshot_image_larger00013.PNG" in cropped_image_path
    os.remove(cropped_image_path)


# @pytest.mark.asyncio
# async def test_handle_autocrop_one(mocker):
#     """Test handle_autocrop_one function."""
#     image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
#     mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))
#     mocker.patch("goob_ai.utils.imgops.cv2.cvtColor", return_value=np.array(Image.open(image_path)))
#     mocker.patch("goob_ai.utils.imgops.cv2.imwrite", return_value=True)
#     mocker.patch("goob_ai.utils.imgops.file_functions.fix_path", return_value=image_path)

#     mock_model = mocker.Mock()
#     mock_model.name = "mock_model"

#     predict_results = (Image.open(image_path), [(0, 0, 100, 100)])

#     cropped_image_path = await handle_autocrop_one(
#         images_filepath=image_path, model=mock_model, predict_results=predict_results
#     )

#     assert cropped_image_path == image_path


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_handle_get_dominant_color_name():
    """Test handle_get_dominant_color function with return_type 'name'."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    urls = [image_path]
    dominant_color = handle_get_dominant_color(urls, return_type="name")

    assert dominant_color == "black"


@pytest.mark.imgops
@pytest.mark.parametrize(
    "r, g, b, expected_hex",
    [
        (255, 0, 0, "#ff0000"),
        (0, 255, 0, "#00ff00"),
        (0, 0, 255, "#0000ff"),
        (255, 255, 255, "#ffffff"),
        (0, 0, 0, "#000000"),
    ],
)
def test_rgb2hex(r, g, b, expected_hex):
    """Test rgb2hex function."""
    from goob_ai.utils.imgops import rgb2hex

    hex_color = rgb2hex(r, g, b)
    assert hex_color == expected_hex


@pytest.mark.imgops
@pytest.mark.parametrize(
    "input_tensor, min_max, expected_output",
    [
        (torch.tensor([-1.0, 0.0, 1.0]), (-1.0, 1.0), torch.tensor([0.0, 0.5, 1.0])),
        (torch.tensor([0.0, 0.5, 1.0]), (0.0, 1.0), torch.tensor([0.0, 0.5, 1.0])),
        (torch.tensor([0.0, 0.5, 1.0]), (0.0, 2.0), torch.tensor([0.0, 0.25, 0.5])),
    ],
)
def test_denorm(input_tensor, min_max, expected_output):
    """Test denorm function with different input ranges."""
    output = denorm(input_tensor, min_max)
    assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_get_pixel_rgb(mocker):
    """Test get_pixel_rgb function."""
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    # mocker.patch("goob_ai.utils.imgops.Image.open", return_value=Image.open(image_path))

    image_pil = Image.open(image_path)
    color = get_pixel_rgb(image_pil)

    assert color == "darkmode"


# @pytest.mark.imgops
# @pytest.mark.parametrize(
#     "input_array, min_max, expected_output",
#     [
#         (np.array([-1.0, 0.0, 1.0]), (-1.0, 1.0), np.array([0.0, 0.5, 1.0])),
#         (np.array([0.0, 0.5, 1.0]), (0.0, 1.0), np.array([0.0, 0.5, 1.0])),
#         (np.array([0.0, 0.5, 1.0]), (0.0, 2.0), np.array([0.0, 0.25, 0.5])),
#     ],
# )
# def test_denorm_numpy(input_array, min_max, expected_output):
#     """Test denorm function with numpy arrays and different input ranges."""
#     output = denorm(input_array, min_max)
#     assert np.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
#     """Test convert_pil_image_to_torch_tensor function (async)."""
#     from goob_ai.utils.imgops import convert_pil_image_to_torch_tensor

#     # Convert the test image to a PIL image
#     async_test_image_pil = Image.fromarray(async_test_image)

#     # Apply convert_pil_image_to_torch_tensor
#     tensor_image = convert_pil_image_to_torch_tensor(async_test_image_pil)

#     # Check if the tensor shape is correct (C, H, W)
#     assert tensor_image.shape == (async_test_image.shape[2], async_test_image.shape[0], async_test_image.shape[1])

#     # Check if the tensor values are in the correct range [0, 1]
#     assert tensor_image.min() >= 0.0
#     assert tensor_image.max() <= 1.0


@pytest.mark.imgops
def test_convert_rgb_to_names():
    """Test convert_rgb_to_names function."""
    from goob_ai.utils.imgops import convert_rgb_to_names

    # Test with a sample RGB tuple
    rgb_tuple = (255, 0, 0)
    color_name = convert_rgb_to_names(rgb_tuple)

    # Check if the color name is correct
    assert color_name == "red"


# @pytest.mark.imgops
# @pytest.mark.asyncio
# async def test_convert_pil_image_to_rgb_channels_async(async_test_image, mocker):
#     """Test convert_pil_image_to_rgb_channels function (async)."""
#     from goob_ai.utils.imgops import convert_pil_image_to_rgb_channels

#     # Mock the get_pil_image_channels function to return 4 channels
#     mocker.patch("goob_ai.utils.imgops.get_pil_image_channels", return_value=4)

#     # Apply convert_pil_image_to_rgb_channels
#     converted_image = convert_pil_image_to_rgb_channels("tests/fixtures/screenshot_image_larger00013.PNG")

#     # Check if the image is converted to RGB
#     assert converted_image.mode == "RGB"  # pyright: ignore[reportAttributeAccessIssue]
#     """Test bgra_to_rgba function (async)."""
#     # Convert the test image to a tensor with an alpha channel
#     test_image_tensor = (
#         torch.from_numpy(np.dstack((async_test_image, np.full(async_test_image.shape[:2], 255))))
#         .permute(2, 0, 1)
#         .float()
#         / 255.0
#     )  # HWC to CHW and normalize

#     # Apply bgra_to_rgba
#     rgba_image_tensor = bgra_to_rgba(test_image_tensor)

#     # Convert back to numpy for comparison
#     rgba_image = rgba_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

#     # Check if the channels are correctly swapped
#     assert np.array_equal(rgba_image[:, :, 0], async_test_image[:, :, 2])  # R channel
#     assert np.array_equal(rgba_image[:, :, 1], async_test_image[:, :, 1])  # G channel
#     assert np.array_equal(rgba_image[:, :, 2], async_test_image[:, :, 0])  # B channel
#     assert np.array_equal(rgba_image[:, :, 3], np.full(async_test_image.shape[:2], 255))  # A channel


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_rgba_to_bgra(test_image):
    """Test rgba_to_bgra function."""
    from goob_ai.utils.imgops import rgba_to_bgra

    # Convert the test image to a tensor with an alpha channel
    test_image_tensor = (
        torch.from_numpy(np.dstack((test_image, np.full(test_image.shape[:2], 255)))).permute(2, 0, 1).float() / 255.0
    )  # HWC to CHW and normalize

    # Apply rgba_to_bgra
    bgra_image_tensor = rgba_to_bgra(test_image_tensor)

    # Convert back to numpy for comparison
    bgra_image = bgra_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(bgra_image[:, :, 0], test_image[:, :, 2])  # B channel
    assert np.array_equal(bgra_image[:, :, 1], test_image[:, :, 1])  # G channel
    assert np.array_equal(bgra_image[:, :, 2], test_image[:, :, 0])  # R channel
    assert np.array_equal(bgra_image[:, :, 3], np.full(test_image.shape[:2], 255))  # A channel


@pytest.mark.imgops
# @pytest.mark.asyncio
def test_rgb_to_bgr_async(test_image):
    """Test rgb_to_bgr function (async)."""
    # Convert the test image to a tensor
    test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

    # Apply rgb_to_bgr
    bgr_image_tensor = rgb_to_bgr(test_image_tensor)

    # Convert back to numpy for comparison
    bgr_image = bgr_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

    # Check if the channels are correctly swapped
    assert np.array_equal(bgr_image[:, :, 0], test_image[:, :, 2])  # B channel
    assert np.array_equal(bgr_image[:, :, 1], test_image[:, :, 1])  # G channel
    assert np.array_equal(bgr_image[:, :, 2], test_image[:, :, 0])  # R channel


# @pytest.mark.asyncio
# async def test_bgr_to_rgb_async(test_image, mocker):
#     """Test bgr_to_rgb function (async)."""
#     # Convert the test image to a tensor
#     test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0  # HWC to CHW and normalize

#     # Apply bgr_to_rgb
#     rgb_image_tensor = bgr_to_rgb(test_image_tensor)

#     # Convert back to numpy for comparison
#     rgb_image = rgb_image_tensor.permute(1, 2, 0).numpy() * 255.0  # CHW to HWC and denormalize

#     # Check if the channels are correctly swapped
#     assert np.array_equal(rgb_image[:, :, 0], test_image[:, :, 2])  # R channel
#     assert np.array_equal(rgb_image[:, :, 1], test_image[:, :, 1])  # G channel
#     assert np.array_equal(rgb_image[:, :, 2], test_image[:, :, 0])  # B channel

#     """Test auto_split_upscale with splitting the image (async)."""
#     scale = 2
#     overlap = 32

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     upscaled_image, depth = auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap)
#     assert upscaled_image.shape == (
#         async_test_image.shape[0] * scale,
#         async_test_image.shape[1] * scale,
#         async_test_image.shape[2],
#     )
#     assert depth > 1


# @pytest.mark.asyncio
# async def test_auto_split_upscale_max_depth_async(async_test_image, mocker):
#     """Test auto_split_upscale with a maximum recursion depth (async)."""
#     scale = 2
#     overlap = 32
#     max_depth = 2

#     # Mock the dummy upscale function to raise an error to force splitting
#     mocker.patch("src.goob_ai.utils.imgops.dummy_upscale_function", side_effect=RuntimeError("Out of memory"))

#     with pytest.raises(RecursionError):
#         auto_split_upscale(async_test_image, dummy_upscale_function, scale, overlap, max_depth=max_depth)


@pytest_asyncio.fixture
async def async_test_image():
    image_path = Path("tests/fixtures/screenshot_image_larger00013.PNG")
    return np.array(Image.open(image_path))


# @pytest.mark.asyncio
def test_auto_split_upscale_no_split_async(test_image):
    """Test auto_split_upscale without splitting the image (async)."""
    scale = 2
    overlap = 0
    upscaled_image, depth = auto_split_upscale(test_image, dummy_upscale_function, scale, overlap)
    assert upscaled_image.shape == (test_image.shape[0], test_image.shape[1], test_image.shape[2])
    assert depth == 1
