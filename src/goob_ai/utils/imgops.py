# pylint: disable=possibly-used-before-assignment
# pylint: disable=consider-using-from-import
# https://github.com/universityofprofessorex/ESRGAN-Bot

# Creative Commons may be contacted at creativecommons.org.
# NOTE: For more examples tqdm + aiofile, search https://github.com/search?l=Python&q=aiofile+tqdm&type=Code
# pylint: disable=no-member

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import gc
import io
import logging
import math
import os
import os.path
import pathlib
import re
import sys
import tempfile
import time
import traceback
import typing
import uuid

from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Tuple

import cv2
import numpy as np
import pytz
import rich
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT

from loguru import logger as LOGGER
from PIL import Image
from scipy.spatial import KDTree
from torch import nn
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import make_grid
from tqdm.auto import tqdm
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

from goob_ai.utils import file_functions
from goob_ai.utils.devices import get_device
from goob_ai.utils.torchutils import load_model


IMG_SIZE_CUTOFF = 1080

TYPE_IMAGE_ARRAY = typing.Union[np.ndarray, typing.Any]

TYPE_SCALE = typing.Union[str, int]

DEVICE = get_device()


class Dimensions(IntEnum):
    HEIGHT = 224
    WIDTH = 224


ImageNdarrayBGR = NewType("ImageBGR", np.ndarray)
ImageNdarrayHWC = NewType("ImageHWC", np.ndarray)
TensorCHW = NewType("TensorCHW", torch.Tensor)

OPENCV_GREEN = (0, 255, 0)
OPENCV_RED = (255, 0, 0)


utc = pytz.utc


def setup_model() -> torch.nn.Module:
    """
    Set up a model for image processing.

    This function loads a pre-trained model for image processing using a predefined
    device and model name. The model is loaded from a file named "ScreenNetV1.pth"
    and is returned as a torch.nn.Module object.

    Returns
    -------
        torch.nn.Module: The loaded image processing model.

    """
    return load_model(DEVICE, model_name="ScreenNetV1.pth")


###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


# SOURCE: https://github.com/langchain-ai/langchain/blob/master/cookbook/multi_modal_QA.ipynb
def encode_image(image_path: str):
    """Getting the base64 string"""

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(resize_base64_image(doc, size=(250, 250)))  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    :param base64_string: A Base64 encoded string of the image to be resized.
    :param size: A tuple representing the new size (width, height) for the image.
    :return: A Base64 encoded string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    return base64.b64encode(buffered.getvalue()).decod


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # img_str = resize_base64_image(img_str, size=(831,623))
    return img_str


def handle_autocrop(
    images_filepaths: list[str],
    cols: int = 5,
    model: Optional[torch.nn.Module] = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
    predict_results: Optional[list[tuple[Image.Image, list[tuple[int, int, int, int]]]]] = None,
) -> list[str]:
    """
    Crop images based on predicted bounding boxes.

    This function takes a list of image file paths and crops each image based on the provided bounding boxes.
    The cropped images are saved to disk, and their file paths are returned.

    Args:
    ----
        images_filepaths (List[str]): List of file paths to the images to be cropped.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (Optional[torch.nn.Module], optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the cropped images. Defaults to False.
        predict_results (Optional[List[Tuple[Image.Image, List[Tuple[int, int, int, int]]]]], optional):
            List of tuples containing images and their corresponding bounding boxes. Defaults to None.

    Returns:
    -------
        List[str]: List of file paths to the cropped images.

    """
    cropped_image_file_paths = []
    for i, image_filepath in enumerate(images_filepaths):
        image, bboxes = predict_results[i]
        # image, bboxes = predict_from_file(image_filepath, model, device)
        img_as_array = np.asarray(image)
        img_as_array = cv2.cvtColor(img_as_array, cv2.COLOR_RGB2BGR)

        xmin_fullsize, ymin_fullsize, xmax_fullsize, ymax_fullsize = bboxes[0]

        startY = int(ymin_fullsize)
        endY = int(ymax_fullsize)
        startX = int(xmin_fullsize)
        endX = int(xmax_fullsize)

        cropped_image = img_as_array[startY:endY, startX:endX]

        # import bpdb
        # bpdb.set_trace()

        image_path_api = pathlib.Path(image_filepath).resolve()
        fname = f"{image_path_api.parent}/cropped-{model.name}-{image_path_api.stem}{image_path_api.suffix}"

        cv2.imwrite(fname, cropped_image)

        cropped_full_path = file_functions.fix_path(fname)

        cropped_image_file_paths.append(cropped_full_path)

    return cropped_image_file_paths


def normalize_rectangle_coords(
    # images_filepath: str,
    # cols: int = 5,
    # model: Optional[torch.nn.Module] = None,
    # device: torch.device = DEVICE,
    # args: Optional[dict] = None,
    # resize: bool = False,
    predict_results=None,
):
    image, bboxes = predict_results
    temp = image.copy()
    img_as_array = np.asarray(temp)
    img_as_array = cv2.cvtColor(img_as_array, cv2.COLOR_RGB2BGR)

    # import bpdb

    # bpdb.set_trace()

    # get fullsize bboxes
    xmin_fullsize, ymin_fullsize, xmax_fullsize, ymax_fullsize = bboxes[0]

    # if we have a negative point to make a rectange with, set it to 0
    startY = max(int(ymin_fullsize), 0)
    endY = max(int(ymax_fullsize), 0)
    startX = max(int(xmin_fullsize), 0)
    endX = max(int(xmax_fullsize), 0)

    rich.print(startY, endY, startX, endX)

    # cropped_image = img_as_array[startY:endY, startX:endX]

    # image_path_api = pathlib.Path(images_filepath).resolve()
    # fname = f"{image_path_api.parent}/cropped-{model.name}-{image_path_api.stem}{image_path_api.suffix}"

    return [startY, endY, startX, endX]


def display_normalized_rectangle(image, out_bbox):
    out_xmin, out_ymin, out_xmax, out_ymax = out_bbox[0]

    out_pt1 = (int(out_xmin), int(out_ymin))
    out_pt2 = (int(out_xmax), int(out_ymax))

    return cv2.rectangle(
        image.squeeze().permute(1, 2, 0).cpu().numpy(),
        out_pt1,
        out_pt2,
        OPENCV_RED,
        2,
    )
    # plt.imshow(out_img)


def handle_autocrop_one(
    images_filepath: str,
    cols: int = 5,
    model: Optional[torch.nn.Module] = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
    predict_results: Optional[tuple[Image.Image, list[tuple[int, int, int, int]]]] = None,
) -> str:
    """
    Crop a single image based on predicted bounding boxes.

    This function takes a single image file path and crops the image based on the provided bounding boxes.
    The cropped image is saved to disk, and its file path is returned.

    Args:
    ----
        images_filepath (str): File path to the image to be cropped.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (Optional[torch.nn.Module], optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the cropped image. Defaults to False.
        predict_results (Optional[Tuple[Image.Image, List[Tuple[int, int, int, int]]]], optional):
            A tuple containing the image and its corresponding bounding boxes. Defaults to None.

    Returns:
    -------
        str: File path to the cropped image.

    """
    image, bboxes = predict_results
    temp = image.copy()
    img_as_array = np.asarray(temp)
    img_as_array = cv2.cvtColor(img_as_array, cv2.COLOR_RGB2BGR)

    # Get fullsize bounding boxes
    xmin_fullsize, ymin_fullsize, xmax_fullsize, ymax_fullsize = bboxes[0]

    # If we have a negative point to make a rectangle with, set it to 0
    startY = max(int(ymin_fullsize), 0)
    endY = max(int(ymax_fullsize), 0)
    startX = max(int(xmin_fullsize), 0)
    endX = max(int(xmax_fullsize), 0)

    rich.print(startY, endY, startX, endX)

    cropped_image = img_as_array[startY:endY, startX:endX]

    image_path_api = pathlib.Path(images_filepath).resolve()
    fname = f"{image_path_api.parent}/cropped-{model.name}-{image_path_api.stem}{image_path_api.suffix}"

    cv2.imwrite(fname, cropped_image)

    return file_functions.fix_path(fname)


def handle_resize(
    images_filepaths: list[str],
    cols: int = 5,
    model: Optional[torch.nn.Module] = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
) -> list[str]:
    """
    Resize a list of images and save them to disk.

    This function takes a list of image file paths, resizes each image if specified,
    and saves the resized images to disk. The file paths of the resized images are returned.

    Args:
    ----
        images_filepaths (List[str]): List of file paths to the images to be resized.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (Optional[torch.nn.Module], optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the images. Defaults to False.

    Returns:
    -------
        List[str]: List of file paths to the resized images.

    """
    resized_image_file_paths = []
    for i, image_filepath in enumerate(images_filepaths):
        image_path_api = pathlib.Path(image_filepath).resolve()
        fname = f"{image_path_api.parent}/cropped-{model.name}-{image_path_api.stem}{image_path_api.suffix}"

        cropped_full_path = file_functions.fix_path(fname)

        if resize:
            to_resize = Image.open(cropped_full_path).convert("RGB")
            resized_pil_image = resize_and_pillarbox(to_resize, 1080, 1350, background=resize)
            if f"{image_path_api.suffix}".lower() == ".png":
                # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
                resized_pil_image.save(fname, optimize=True, compress_level=9)
            elif f"{image_path_api.suffix}".lower() == (".jpg" or ".jpeg"):
                resized_pil_image.save(fname, quality="web_medium")

        resized_image_file_paths.append(cropped_full_path)

    return resized_image_file_paths


def handle_resize_one(
    images_filepath: str,
    cols: int = 5,
    model: Optional[torch.nn.Module] = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
) -> str:
    """
    Resize a single image and save it to disk.

    This function takes a single image file path, resizes the image if specified,
    and saves the resized image to disk. The file path of the resized image is returned.

    Args:
    ----
        images_filepath (str): File path to the image to be resized.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (Optional[torch.nn.Module], optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the image. Defaults to False.

    Returns:
    -------
        str: File path to the resized image.

    """
    image_path_api = pathlib.Path(images_filepath).resolve()
    fname = f"{image_path_api.parent}/cropped-{model.name}-{image_path_api.stem}{image_path_api.suffix}"

    cropped_full_path = file_functions.fix_path(fname)

    if resize:
        to_resize = Image.open(cropped_full_path).convert("RGB")
        resized_pil_image = resize_and_pillarbox(to_resize, 1080, 1350, background=resize)
        if f"{image_path_api.suffix}".lower() == ".png":
            # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
            resized_pil_image.save(fname, optimize=True, compress_level=9)
        elif f"{image_path_api.suffix}".lower() == (".jpg" or ".jpeg"):
            resized_pil_image.save(fname, quality="web_medium")

    return cropped_full_path


def handle_predict(
    images_filepaths: list[str],
    cols: int = 5,
    model: Optional[torch.nn.Module] = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
) -> list:
    # -> List[Tuple[Image, torch.Tensor]]:
    # -> List[Tuple[Image.Image, List[Tuple[int, int, int, int]]]]:
    """
    Predict bounding boxes for a list of images.

    This function takes a list of image file paths and uses a model to predict bounding boxes for each image.
    The predicted bounding boxes and the corresponding images are returned as a list of tuples.

    Args:
    ----
        images_filepaths (List[str]): List of file paths to the images for prediction.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (Optional[torch.nn.Module], optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the images. Defaults to False.

    Returns:
    -------
        List[Tuple[Image.Image, List[Tuple[int, int, int, int]]]]:
            List of tuples containing images and their corresponding bounding boxes.

    """
    image_and_bboxes_list = []
    for image_filepath in images_filepaths:
        image, bboxes = predict_from_file(image_filepath, model, device)
        image_and_bboxes_list.append([image, bboxes])
    return image_and_bboxes_list


def handle_predict_one(
    images_filepath: str,
    cols: int = 5,
    model: torch.nn.Module | None = None,
    device: torch.device = DEVICE,
    args: Optional[dict] = None,
    resize: bool = False,
):
    #  -> Tuple[Image.Image, torch.Tensor]:
    """
    Predict bounding boxes for a single image.

    This function takes a single image file path and uses a model to predict bounding boxes for the image.
    The predicted bounding boxes and the corresponding image are returned as a tuple.

    Args:
    ----
        images_filepath (str): File path to the image for prediction.
        cols (int, optional): Number of columns for display purposes. Defaults to 5.
        model (torch.nn.Module | None, optional): The model used for prediction. Defaults to None.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.
        args (Optional[dict], optional): Additional arguments for the function. Defaults to None.
        resize (bool, optional): Whether to resize the image. Defaults to False.

    Returns:
    -------
        Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
            A tuple containing the image and its corresponding bounding boxes.

    """
    assert cols
    image, bboxes = predict_from_file(images_filepath, model, device)
    return image, bboxes


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################


def resize_image_and_bbox(
    image: torch.Tensor,
    boxes: torch.Tensor,
    dims: tuple[int, int] = (300, 300),
    return_percent_coords: bool = False,
    device: torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resize an image and its bounding boxes.

    This function resizes an image to the specified dimensions and adjusts the bounding boxes accordingly.
    It can return the bounding boxes as either absolute coordinates or percent coordinates.

    Args:
    ----
        image (torch.Tensor): The input image tensor.
        boxes (torch.Tensor): The bounding boxes tensor with dimensions (n_objects, 4).
        dims (Tuple[int, int], optional): The target dimensions for resizing. Defaults to (300, 300).
        return_percent_coords (bool, optional): Whether to return bounding boxes as percent coordinates. Defaults to False.
        device (torch.device, optional): The device to perform the operation on. Defaults to DEVICE.

    Returns:
    -------
        Tuple[torch.Tensor, torch.Tensor]: The resized image tensor and the updated bounding boxes tensor.

    """
    image_tensor_to_resize_height = image.shape[1]
    image_tensor_to_resize_width = image.shape[2]

    # Resize image
    new_image = FT.resize(image, dims, InterpolationMode.BILINEAR, antialias=True)

    # Resize bounding boxes
    old_dims = (
        torch.FloatTensor(
            [
                image_tensor_to_resize_width,
                image_tensor_to_resize_height,
                image_tensor_to_resize_width,
                image_tensor_to_resize_height,
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0).to(device)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


# SOURCE: https://www.learnpytorch.io/09_pytorch_model_deployment/
def pred_and_store(
    paths: list[pathlib.Path],
    model: torch.nn.Module,
    device: torch.device = DEVICE,
) -> list[dict[str, Any]]:
    """
    Predict bounding boxes for images and store the results.

    This function loops through a list of image paths, performs predictions using the provided model,
    and stores the prediction information in a list of dictionaries. Each dictionary contains the image path,
    prediction bounding boxes, and other relevant information.

    Args:
    ----
        paths (List[pathlib.Path]): List of image paths to process.
        model (torch.nn.Module): The model used for prediction.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.

    Returns:
    -------
        List[Dict[str, Any]]: A list of dictionaries containing prediction information for each image.

    """
    for path in tqdm(paths):
        pred_dict = {"image_path": path}

        targetSize = Dimensions.HEIGHT

        img: ImageNdarrayBGR  # type: ignore

        img_channel: int
        img_height: int
        img_width: int

        # import bpdb
        # bpdb.set_trace()

        img, img_channel, img_height, img_width = read_image_to_bgr(f"{paths[0]}")

        resized = cv2.resize(img, (targetSize, targetSize), interpolation=cv2.INTER_AREA)
        print(resized.shape)

        # normalize and change output to (c, h, w)
        resized_tensor: torch.Tensor = torch.from_numpy(resized).permute(2, 0, 1) / 255.0

        # import bpdb
        # bpdb.set_trace()

        model.to(device)
        model.eval()

        with torch.inference_mode():
            # Convert to (bs, c, h, w)
            unsqueezed_tensor = resized_tensor.unsqueeze(0).to(device)

            # predict
            out_bbox: torch.Tensor = model(unsqueezed_tensor)

            xmin, ymin, xmax, ymax = out_bbox[0]
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))

            starting_point = pt1
            end_point = pt2
            color = (255, 0, 0)
            thickness = 2

            out_img = cv2.rectangle(
                unsqueezed_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8"),
                starting_point,
                end_point,
                color,
                thickness,
            )

            image_tensor_to_resize = resized_tensor
            resized_bboxes_tensor = out_bbox[0]
            resized_height = img_height
            resized_width = img_width
            resized_dims = (resized_height, resized_width)

            image_tensor_to_resize.shape[0]
            image_tensor_to_resize.shape[1]
            image_tensor_to_resize.shape[2]

            fullsize_image, fullsize_bboxes = resize_image_and_bbox(
                image_tensor_to_resize,
                resized_bboxes_tensor,
                dims=resized_dims,
                return_percent_coords=False,
                device=device,
            )

            (
                xmin_fullsize,
                ymin_fullsize,
                xmax_fullsize,
                ymax_fullsize,
            ) = fullsize_bboxes[0]

            (int(xmin_fullsize), int(ymin_fullsize))
            (int(xmax_fullsize), int(ymax_fullsize))

            color = OPENCV_RED
            thickness = 1

    print(fullsize_bboxes)

    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    LOGGER.info(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")

    return fullsize_bboxes


def get_pil_image_channels(image_path: str) -> int:
    """
    Open an image and get the number of channels it has.

    This function loads an image using the Pillow library and converts it to a tensor.
    It then returns the number of channels in the image.

    Args:
    ----
        image_path (str): The path to the image file.

    Returns:
    -------
        int: The number of channels in the image.

    """
    # load pillow image
    pil_img = Image.open(
        image_path
    )  # 'pil_img': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1179x2556 at 0x32AA189D0>,
    # import bpdb

    # bpdb.set_trace()

    # PILToTensor: Convert a PIL Image to a tensor of the same type - this does not scale values.
    # This transform does not support torchscript.
    # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    transform = transforms.Compose([transforms.PILToTensor()])

    ########################################################################
    # >>> type(transform)
    # <class 'torchvision.transforms.transforms.Compose'>
    #  dtype=torch.uint8
    #  shape = torch.Size([3, 2556, 1179])
    # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    ########################################################################
    img_tensor: torch.Tensor = transform(pil_img)

    # # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    # pil_img_tensor = transforms.PILToTensor()(pil_img)

    return img_tensor.shape[0]


def convert_pil_image_to_rgb_channels(image_path: str) -> Image:
    """
    Convert a PIL image to have the appropriate number of color channels.

    This function checks the number of channels in a PIL image and converts it to RGB if it has a different number of channels.

    Args:
    ----
        image_path (str): The path to the image file.

    Returns:
    -------
        Image: The converted PIL image with RGB channels.

    """
    return Image.open(image_path).convert("RGB") if get_pil_image_channels(image_path) != 4 else Image.open(image_path)


def read_image_to_bgr(image_path: str) -> tuple[np.ndarray, int, int, int]:
    """
    Read an image from the specified file path and convert it to BGR format.

    This function reads an image from the given file path using OpenCV, converts it from BGR to RGB format,
    and returns the image array along with its number of channels, height, and width.

    Args:
    ----
        image_path (str): The path to the image file.

    Returns:
    -------
        Tuple[np.ndarray, int, int, int]: A tuple containing the image array in BGR format,
                                          the number of channels, the height, and the width of the image.

    Raises:
    ------
        FileNotFoundError: If the image file does not exist at the specified path.

    Example:
    -------
        >>> image, channels, height, width = read_image_to_bgr("path/to/image.jpg")
        >>> print(image.shape)
        (height, width, channels)

    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_channel = image.shape[2]
    img_height = image.shape[0]
    img_width = image.shape[1]
    return image, img_channel, img_height, img_width


def convert_image_from_hwc_to_chw(img: ImageNdarrayBGR) -> torch.Tensor:
    """
    Convert an image from HWC (Height, Width, Channels) format to CHW (Channels, Height, Width) format.

    This function takes an image in HWC format and converts it to CHW format, which is commonly used in
    deep learning frameworks like PyTorch.

    Args:
    ----
        img (ImageNdarrayBGR): The input image in HWC format.

    Returns:
    -------
        torch.Tensor: The output image in CHW format.

    """
    img: torch.Tensor = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # (h,w,c) -> (c,h,w)
    return img


# convert image back and forth if needed: https://stackoverflow.com/questions/68207510/how-to-use-torchvision-io-read-image-with-image-as-variable-not-stored-file
def convert_pil_image_to_torch_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image to a PyTorch tensor.

    This function takes a PIL image and converts it to a PyTorch tensor.
    The resulting tensor will have its channels in the order expected by PyTorch (C x H x W).

    Args:
    ----
        pil_image (Image.Image): The input image in PIL format.

    Returns:
    -------
        torch.Tensor: The converted image as a PyTorch tensor.

    """
    # NOTE: https://github.com/ultralytics/ultralytics/issues/9185
    return FT.to_tensor(pil_image).half()


# convert image back and forth if needed: https://stackoverflow.com/questions/68207510/how-to-use-torchvision-io-read-image-with-image-as-variable-not-stored-file
def convert_tensor_to_pil_image(tensor_image: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to a PIL image.

    This function takes a PyTorch tensor and converts it to a PIL image.
    The input tensor is expected to have its channels in the order expected by PyTorch (C x H x W).

    Args:
    ----
        tensor_image (torch.Tensor): The input image as a PyTorch tensor.

    Returns:
    -------
        Image.Image: The converted image in PIL format.

    """
    return FT.to_pil_image(tensor_image)


def predict_from_file(
    path_to_image_from_cli: str,
    model: torch.nn.Module,
    device: torch.device = DEVICE,
):
    # -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Perform predictions on an individual image file.

    This function takes the path to an image file, loads the image, and uses the provided model
    to predict bounding boxes for objects in the image. The function returns the image and the
    predicted bounding boxes.

    Args:
    ----
        path_to_image_from_cli (str): The file path to the image for prediction.
        model (torch.nn.Module): The model used for prediction.
        device (torch.device, optional): The device to run the model on. Defaults to DEVICE.

    Returns:
    -------
        Tuple[Image.Image, List[Tuple[int, int, int, int]]]: A tuple containing the image and a list of bounding boxes.

    """
    # ic(f"Predict | individual file {path_to_image_from_cli} ...")
    LOGGER.info(f"Predict | individual file {path_to_image_from_cli} ...")
    image_path_api = pathlib.Path(path_to_image_from_cli).resolve()
    LOGGER.info(image_path_api)
    # ic(image_path_api)

    paths = [image_path_api]
    img = convert_pil_image_to_rgb_channels(f"{paths[0]}")

    bboxes = pred_and_store(paths, model, device=device)

    return img, bboxes


def get_pixel_rgb(image_pil: Image) -> str:
    """
    Get the color of the first pixel in an image.

    This function retrieves the RGB values of the first pixel (at position (1, 1)) in the provided PIL image.
    It then determines if the color is white or dark mode based on the RGB values.

    Args:
    ----
        image_pil (Image): The input image in PIL format.

    Returns:
    -------
        str: A string indicating the color of the first pixel, either "white" or "darkmode".

    """
    r, g, b = image_pil.getpixel((1, 1))  # pyright: ignore[reportAttributeAccessIssue]

    color = "white" if (r, g, b) == (255, 255, 255) else "darkmode"
    print(f"GOT COLOR {color} -- {r},{g},{b}")
    return color


def resize_and_pillarbox(image_pil: Image.Image, width: int, height: int, background: bool = True) -> Image.Image:
    """
    Resize a PIL image while maintaining its aspect ratio and adding pillarbox.

    This function resizes a PIL image to fit within the specified width and height while maintaining
    the original aspect ratio. It adds pillarbox (padding) to the image to fill the remaining space
    with a specified background color.

    Args:
    ----
        image_pil (Image.Image): The input image in PIL format.
        width (int): The target width for the resized image.
        height (int): The target height for the resized image.
        background (str, optional): The background color for the pillarbox. Defaults to "white".

    Returns:
    -------
        Image.Image: The resized image with pillarbox added.

    Example:
    -------
        >>> image = Image.open("path/to/image.jpg")
        >>> resized_image = resize_and_pillarbox(image, 1080, 1350, background="white")
        >>> resized_image.show()

    """
    autodetect_background = get_pixel_rgb(image_pil)

    ratio_w = width / image_pil.width  # pyright: ignore[reportAttributeAccessIssue]
    ratio_h = height / image_pil.height  # pyright: ignore[reportAttributeAccessIssue]
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)  # pyright: ignore[reportAttributeAccessIssue]
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.Resampling.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]
    if background and autodetect_background == "white":
        background = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    elif background and autodetect_background == "darkmode":
        background = Image.new("RGBA", (width, height), (22, 32, 42, 1))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)  # pyright: ignore[reportAttributeAccessIssue]
    return background.convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]


def convert_rgb_to_names(rgb_tuple: tuple[int, int, int]) -> str:
    # a dictionary of all the hex and their respective names in css3
    css3_db = CSS3_HEX_TO_NAMES
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))

    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    rich.print(f"closest match: {names[index]}")
    return f"{names[index]}"


def get_all_corners_color(urls: list[str]) -> dict[str, str]:
    """
    Get the colors of the four corners of images.

    This function opens each image from the provided URLs, converts them to RGB,
    and retrieves the colors of the top-left, top-right, bottom-left, and bottom-right corners.

    Args:
    ----
        urls (List[str]): A list of URLs pointing to the image files.

    Returns:
    -------
        Dict[str, str]: A dictionary containing the colors of the four corners of the images.

    """
    pbar = tqdm(urls)
    pixel_data = {
        "top_left": "",
        "top_right": "",
        "bottom_left": "",
        "bottom_right": "",
    }
    for url in pbar:
        img = Image.open(url).convert("RGB")
        width, height = img.size
        pixel_layout = img.load()
        pixel_data["top_left"] = top_left = pixel_layout[0, 0]
        pixel_data["top_right"] = top_right = pixel_layout[width - 1, 0]
        pixel_data["bottom_left"] = bottom_left = pixel_layout[0, height - 1]
        pixel_data["bottom_right"] = bottom_right = pixel_layout[width - 1, height - 1]
        rich.print(pixel_data)
    return pixel_data


def rgb2hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to a hexadecimal color string.

    This function takes the red, green, and blue components of a color
    and converts them to a hexadecimal string representation.

    Args:
    ----
        r (int): The red component of the color, in the range [0, 255].
        g (int): The green component of the color, in the range [0, 255].
        b (int): The blue component of the color, in the range [0, 255].

    Returns:
    -------
        str: The hexadecimal string representation of the color, prefixed with '#'.

    Example:
    -------
        >>> rgb2hex(255, 0, 0)
        '#ff0000'

    """
    LOGGER.info(f"RGB2HEX: {r} {g} {b}")
    return f"#{r:02x}{g:02x}{b:02x}"
    # LOGGER.info(f"TYPE RGB2HEX: {type(r)} {type(g)} {type(b)}")
    # return "#{:02x}{:02x}{:02x}".format(r, g, b)


def handle_get_dominant_color(urls: list[str], return_type: str = "name") -> str:
    """
    Get the dominant color from the corners of images.

    This function retrieves the colors of the four corners of images from the provided URLs.
    It then determines the dominant color based on the corner colors and returns it either as a color name or hex value.

    Args:
    ----
        urls (List[str]): A list of URLs pointing to the image files.
        return_type (str, optional): The format to return the dominant color.
            Can be "name" for color name or "hex" for hex value. Defaults to "name".

    Returns:
    -------
        str: The dominant color in the specified format (name or hex).

    """
    start_time = time.time()
    corner_pixels = get_all_corners_color(urls)

    if (
        corner_pixels["top_left"]
        == corner_pixels["top_right"]
        == corner_pixels["bottom_left"]
        == corner_pixels["bottom_right"]
    ):
        r, g, b = corner_pixels["top_left"]
    else:
        r, g, b = corner_pixels["top_right"]
    background_color = rgb2hex(r, g, b)
    rich.print(background_color)
    color_name = convert_rgb_to_names((r, g, b))
    rich.print(color_name)

    duration = time.time() - start_time
    print(f"Calculated 1 image in {duration} seconds")
    if return_type == "name":
        return color_name
    elif return_type == "hex":
        return background_color


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a BGR image to RGB.

    This function takes an image tensor in BGR format and converts it to RGB format by flipping the color channels.

    Args:
    ----
        image (torch.Tensor): The input image tensor in BGR format.

    Returns:
    -------
        torch.Tensor: The output image tensor in RGB format.

    """
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    # same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)


def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a BGRA image to RGBA.

    This function takes an image tensor in BGRA format and converts it to RGBA format
    by rearranging the color channels.

    Args:
    ----
        image (torch.Tensor): The input image tensor in BGRA format.

    Returns:
    -------
        torch.Tensor: The output image tensor in RGBA format.

    """
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out


def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    """
    Convert a RGBA image to BGRA format.

    This function takes an image tensor in RGBA format and converts it to BGRA format
    by rearranging the color channels.

    Args:
    ----
        image (torch.Tensor): The input image tensor in RGBA format.

    Returns:
    -------
        torch.Tensor: The output image tensor in BGRA format.

    """
    # same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)


def denorm(x: torch.Tensor | np.ndarray, min_max: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor | np.ndarray:
    """
    Denormalize a tensor or numpy array from a specified range to [0, 1].

    This function converts values from a given range (default is [-1, 1]) to the range [0, 1].
    It is useful for reversing normalization applied during preprocessing.

    Args:
    ----
        x (torch.Tensor | np.ndarray): The input tensor or numpy array to be denormalized.
        min_max (tuple[float, float], optional): The range of the input values. Defaults to (-1.0, 1.0).

    Returns:
    -------
        torch.Tensor | np.ndarray: The denormalized tensor or numpy array.

    Raises:
    ------
        TypeError: If the input is not a torch.Tensor or np.ndarray.

    """
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError(
            "Got unexpected object type, expected torch.Tensor or \
        np.ndarray"
        )


def norm(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Normalize a tensor or numpy array from [0, 1] range to [-1, 1] range.

    This function converts values from the range [0, 1] to the range [-1, 1].
    It is useful for normalizing data before feeding it into a neural network.

    Args:
    ----
        x (torch.Tensor | np.ndarray): The input tensor or numpy array to be normalized.

    Returns:
    -------
        torch.Tensor | np.ndarray: The normalized tensor or numpy array.

    Raises:
    ------
        TypeError: If the input is not a torch.Tensor or np.ndarray.

    """
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError(
            "Got unexpected object type, expected torch.Tensor or \
        np.ndarray"
        )


async def np2tensor(
    img: np.ndarray,
    bgr2rgb: bool = True,
    data_range: float = 1.0,
    normalize: bool = False,
    change_range: bool = True,
    add_batch: bool = True,
) -> torch.Tensor:
    """
    Convert a numpy image array into a PyTorch tensor.

    This function converts a numpy image array into a PyTorch tensor. It supports
    various options such as converting BGR to RGB, normalizing the image, changing
    the data range, and adding a batch dimension.

    Args:
    ----
        img (np.ndarray): The input image as a numpy array.
        bgr2rgb (bool, optional): Whether to convert BGR to RGB. Defaults to True.
        data_range (float, optional): The data range for the image. Defaults to 1.0.
        normalize (bool, optional): Whether to normalize the image. Defaults to False.
        change_range (bool, optional): Whether to change the data range. Defaults to True.
        add_batch (bool, optional): Whether to add a batch dimension. Defaults to True.

    Returns:
    -------
        torch.Tensor: The converted image as a PyTorch tensor.

    Raises:
    ------
        TypeError: If the input is not a numpy array.

    """
    if not isinstance(img, np.ndarray):  # images expected to be uint8 -> 255
        raise TypeError("Got unexpected object type, expected np.ndarray")
    # check how many channels the image has, then condition, like in my BasicSR. ie. RGB, RGBA, Gray
    # if bgr2rgb:
    # img = img[:, :, [2, 1, 0]] #BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo
        elif np.issubdtype(img.dtype, np.floating):
            info = np.finfo
        img = img * data_range / info(img.dtype).max  # uint8 = /255
    img = torch.from_numpy(
        np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    ).float()  # "HWC to CHW" and "numpy to tensor"
    if bgr2rgb:
        if img.shape[0] == 3:  # RGB
            # BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img = bgr_to_rgb(img)
        elif img.shape[0] == 4:  # RGBA
            # BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
            img = bgra_to_rgba(img)
    if add_batch:
        # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
        img.unsqueeze_(0)
    if normalize:
        img = norm(img)
    return img


# 2np


async def tensor2np(
    img: torch.Tensor,
    rgb2bgr: bool = True,
    remove_batch: bool = True,
    data_range: int = 255,
    denormalize: bool = False,
    change_range: bool = True,
    imtype: type = np.uint8,
) -> np.ndarray:
    """
    Convert a Tensor array into a numpy image array.

    This function converts a PyTorch tensor into a numpy image array. It supports
    various options such as converting RGB to BGR, removing the batch dimension,
    changing the data range, and denormalizing the image.

    Args:
    ----
        img (torch.Tensor): The input image tensor array. It can be 4D (B, (3/1), H, W),
            3D (C, H, W), or 2D (H, W), with any range and RGB channel order.
        rgb2bgr (bool, optional): Whether to convert RGB to BGR. Defaults to True.
        remove_batch (bool, optional): Whether to remove the batch dimension if the tensor
            is of shape BCHW. Defaults to True.
        data_range (int, optional): The data range for the image. Defaults to 255.
        denormalize (bool, optional): Whether to denormalize the image from [-1, 1] range
            back to [0, 1]. Defaults to False.
        change_range (bool, optional): Whether to change the data range. Defaults to True.
        imtype (type, optional): The desired type of the converted numpy array. Defaults to np.uint8.

    Returns:
    -------
        np.ndarray: The converted image as a numpy array. It will be 3D (H, W, C) or 2D (H, W),
            with values in the range [0, 255] and of type np.uint8 (default).

    Raises:
    ------
        TypeError: If the input is not a torch.Tensor.

    Example:
    -------
        >>> tensor_image = torch.randn(1, 3, 256, 256)
        >>> numpy_image = await tensor2np(tensor_image)
        >>> print(numpy_image.shape)
        (256, 256, 3)

    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Got unexpected object type, expected torch.Tensor")
    n_dim = img.dim()

    # TODO: Check: could denormalize here in tensor form instead, but end result is the same

    img = img.float().cpu()

    if n_dim in [4, 3]:
        # if n_dim == 4, has to convert to 3 dimensions, either removing batch or by creating a grid
        if n_dim == 4 and remove_batch:
            if img.shape[0] > 1:
                # leave only the first image in the batch
                img = img[0, ...]
            else:
                # remove a fake batch dimension
                img = img.squeeze()
                # squeeze removes batch and channel of grayscale images (dimensions = 1)
                if len(img.shape) < 3:
                    # add back the lost channel dimension
                    img = img.unsqueeze(dim=0)
        # convert images in batch (BCHW) to a grid of all images (C B*H B*W)
        else:
            n_img = len(img)
            img = make_grid(img, nrow=int(math.sqrt(n_img)), normalize=False)

        if img.shape[0] == 3 and rgb2bgr:  # RGB
            # RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr:  # RGBA
            # RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # "CHW to HWC" -> # HWC, BGR
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}")

    # if rgb2bgr:
    # img_np = img_np[[2, 1, 0], :, :] #RGB to BGR -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    # TODO: Check: could denormalize in the begining in tensor form instead
    if denormalize:
        img_np = denorm(img_np)  # denormalize if needed
    if change_range:
        # clip to the data_range
        img_np = np.clip(data_range * img_np, 0, data_range).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    # has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)


def auto_split_upscale(
    lr_img: np.ndarray,
    upscale_function: typing.Callable[[np.ndarray], np.ndarray],
    scale: int = 4,
    overlap: int = 32,
    max_depth: Optional[int] = None,
    current_depth: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Recursively upscale an image by splitting it into smaller sections.

    This function attempts to upscale an image using the provided `upscale_function`.
    If the upscaling process runs out of memory, the image is split into four quadrants,
    and the upscaling is attempted on each quadrant recursively.

    Args:
    ----
        lr_img (np.ndarray): The low-resolution input image.
        upscale_function (typing.Callable[[np.ndarray], np.ndarray]): The function to upscale the image.
        scale (int, optional): The scaling factor. Defaults to 4.
        overlap (int, optional): The overlap between image sections to avoid seams. Defaults to 32.
        max_depth (Optional[int], optional): The maximum recursion depth. If None, recursion continues until out of memory. Defaults to None.
        current_depth (int, optional): The current recursion depth. Defaults to 1.

    Returns:
    -------
        typing.Tuple[np.ndarray, int]: The upscaled image and the depth of recursion used.

    """
    if current_depth > 1 and (lr_img.shape[0] == lr_img.shape[1] == overlap):
        raise RecursionError("Reached bottom of recursion depth.")

    # Attempt to upscale if unknown depth or if reached known max depth
    if max_depth is None or max_depth == current_depth:
        try:
            result = upscale_function(lr_img)
            return result, current_depth
        except RuntimeError as e:
            if "allocate" not in str(e):
                raise RuntimeError(e) from e

            # Collect garbage (clear VRAM)
            torch.cuda.empty_cache()
            gc.collect()
    h, w, c = lr_img.shape

    # Split image into 4ths
    top_left = lr_img[: h // 2 + overlap, : w // 2 + overlap, :]
    top_right = lr_img[: h // 2 + overlap, w // 2 - overlap :, :]
    bottom_left = lr_img[h // 2 - overlap :, : w // 2 + overlap, :]
    bottom_right = lr_img[h // 2 - overlap :, w // 2 - overlap :, :]

    # Recursively upscale the quadrants
    # After we go through the top left quadrant, we know the maximum depth and no longer need to test for out-of-memory
    top_left_rlt, depth = auto_split_upscale(
        top_left,
        upscale_function,
        scale=scale,
        overlap=overlap,
        current_depth=current_depth + 1,
    )
    top_right_rlt, _ = auto_split_upscale(
        top_right,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )
    bottom_left_rlt, _ = auto_split_upscale(
        bottom_left,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )
    bottom_right_rlt, _ = auto_split_upscale(
        bottom_right,
        upscale_function,
        scale=scale,
        overlap=overlap,
        max_depth=depth,
        current_depth=current_depth + 1,
    )

    # Define output shape
    out_h = h * scale
    out_w = w * scale

    # Create blank output image
    output_img = np.zeros((out_h, out_w, c), np.uint8)

    # Fill output image with tiles, cropping out the overlaps
    output_img[: out_h // 2, : out_w // 2, :] = top_left_rlt[: out_h // 2, : out_w // 2, :]
    output_img[: out_h // 2, -out_w // 2 :, :] = top_right_rlt[: out_h // 2, -out_w // 2 :, :]
    output_img[-out_h // 2 :, : out_w // 2, :] = bottom_left_rlt[-out_h // 2 :, : out_w // 2, :]
    output_img[-out_h // 2 :, -out_w // 2 :, :] = bottom_right_rlt[-out_h // 2 :, -out_w // 2 :, :]

    return output_img, depth


async def aio_main():
    # Load the test image
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    img = Image.open(image_path)
    test_image = np.array(img)

    # Test with default parameters
    tensor_image = await np2tensor(test_image)
    # import bpdb

    # bpdb.set_trace()
    assert isinstance(tensor_image, torch.Tensor)
    assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])
    await LOGGER.complete()


def main():
    asyncio.run(aio_main())


if __name__ == "__main__":
    # import pathlib
    # from goob_ai.utils.imgops import pred_and_store, setup_model

    # # from goob_ai.utils.imgops import load_model, setup_model
    # model = setup_model()

    # # Mock the model to return dummy bounding boxes
    # # mock_model = mocker.Mock()
    # # mock_model.return_value = torch.tensor([[0, 0, 100, 100]])

    # # Call the pred_and_store function
    # paths = [pathlib.Path("tests/fixtures/screenshot_image_larger00013.PNG")]
    # result = pred_and_store(paths, model)
    # print(result)

    # from goob_ai.utils.imgops import np2tensor

    # # Load the test image
    # image_path = "tests/fixtures/screenshot_image_larger00013.PNG"
    # test_image = np.array(Image.open(image_path))

    # # Test with default parameters
    # tensor_image = await np2tensor(test_image)
    # assert isinstance(tensor_image, torch.Tensor)
    # assert tensor_image.shape == (1, test_image.shape[2], test_image.shape[0], test_image.shape[1])
    # main()
    image_path = "tests/fixtures/screenshot_image_larger00013.PNG"

    model = setup_model()
    # mock_model = mocker.Mock()
    # mock_model.name = "mock_model"

    predict_results = handle_predict(images_filepaths=[image_path], model=model)

    # import bpdb

    # bpdb.set_trace()
    assert isinstance(predict_results, list)
