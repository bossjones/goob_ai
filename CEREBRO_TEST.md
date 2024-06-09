```python
from cerebro_bot.utils.arch.ScreenCropNet import ObjLocModel as ScreenCropNet_ObjLocModel
from cerebro_bot.aio_settings import aiosettings
import torch
import pathlib
import rich
from torch import nn
from cerebro_bot import debugger
import cv2
import numpy as np
import asyncio
import concurrent.futures
from enum import IntEnum
import functools
import logging
import os
import os.path
import pathlib
import sys
import tempfile
from timeit import default_timer as timer
import traceback
import typing
from typing import Dict, List, NewType, Optional

from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import torchvision.transforms.functional as pytorch_transforms_functional
from tqdm.auto import tqdm


IMG_SIZE_CUTOFF = 1080

TYPE_IMAGE_ARRAY = typing.Union[np.ndarray, typing.Any]

TYPE_SCALE = typing.Union[str, int]

CUDA_AVAILABLE = torch.cuda.is_available()  # True


class Dimensions(IntEnum):
    HEIGHT = 224
    WIDTH = 224


ImageNdarrayBGR = NewType("ImageBGR", np.ndarray)
ImageNdarrayHWC = NewType("ImageHWC", np.ndarray)
TensorCHW = NewType("TensorCHW", torch.Tensor)

OPENCV_GREEN = (0, 255, 0)
OPENCV_RED = (255, 0, 0)


def read_image_to_bgr(image_path: str) -> ImageNdarrayBGR:
    """Read the image from image id.

    returns ImageNdarrayBGR.

    Opencv returns ndarry in format = row (height) x column (width) x color (3)
    """

    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image /= 255.0  # Normalize

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # import bpdb
    # bpdb.set_trace()
    # img_shape = image.shape
    img_channel = image.shape[2]
    img_height = image.shape[0]
    img_width = image.shape[1]
    return image, img_channel, img_height, img_width

def predict_from_file(path_to_image_from_cli: str, model: torch.nn.Module, device: torch.device):
    """wrapper function to perform predictions on individual files

    Args:
        path_to_image_from_cli (str): eg.  "/Users/malcolm/Downloads/2020-11-25_10-47-32_867.jpeg
        model (torch.nn.Module): _description_
        transform (torchvision.transforms): _description_
        class_names (List[str]): _description_
        device (torch.device): _description_
        args (argparse.Namespace): _description_
    """
    # ic(f"Predict | individual file {path_to_image_from_cli} ...")
    image_path_api = pathlib.Path(path_to_image_from_cli).resolve()
    # ic(image_path_api)

    paths = [image_path_api]
    img = convert_pil_image_to_rgb_channels(f"{paths[0]}")

    bboxes = pred_and_store(paths, model, device=device)

    return img, bboxes

def handle_predict_one(
    images_filepath: str,
    cols=5,
    model=None,
    device=None,
    args=None,
    resize=False,
):
    assert cols
    # image_and_bboxes_list = []
    # for i, image_filepath in enumerate(images_filepaths):
    image, bboxes = predict_from_file(images_filepath, model, device)
    # image_and_bboxes_list.append()
    return image, bboxes

# NOTE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
def load_model_for_inference(save_path: str, device: str) -> nn.Module:
    model = ScreenCropNet_ObjLocModel()
    model.name = "ObjLocModelV1"
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Model loaded from path {save_path} successfully.")
    # Get the model size in bytes then convert to megabytes
    model_size = pathlib.Path(save_path).stat().st_size // (1024 * 1024)
    print(f"{save_path} | feature extractor model size: {model_size} MB")
    return model


# wrapper function of common code
def run_get_model_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    path_to_model: pathlib.PosixPath,
) -> torch.nn.Module:
    """wrapper function to load model .pth file from disk

    Args:
        model (torch.nn.Module): _description_
        device (torch.device): _description_
        class_names (List[str]): _description_

    Returns:
        Tuple[pathlib.PosixPath, torch.nn.Module]: _description_
    """
    return load_model_for_inference(path_to_model, device)



def convert_pil_image_to_rgb_channels(image_path: str):
    """Convert Pil image to have the appropriate number of color channels

    Args:
        image_path (str): _description_

    Returns:
        _type_: _description_
    """
    return Image.open(image_path).convert("RGB") if get_pil_image_channels(image_path) != 4 else Image.open(image_path)

def get_pil_image_channels(image_path: str) -> int:
    """Open an image and get the number of channels it has.

    Args:
        image_path (str): _description_

    Returns:
        int: _description_
    """
    # load pillow image
    pil_img = Image.open(image_path)

    # Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    pil_img_tensor = transforms.PILToTensor()(pil_img)

    return pil_img_tensor.shape[0]


# SOURCE: https://www.learnpytorch.io/09_pytorch_model_deployment/
# 1. Create a function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time
def pred_and_store(
    paths: List[pathlib.Path],
    model: torch.nn.Module,
    # transform: torchvision.transforms,
    # class_names: List[str],
    device: torch.device = "",
) -> List[Dict]:
    # 3. Loop through target paths
    for path in tqdm(paths):
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {"image_path": path}

        # 6. Start the prediction timer
        timer()

        targetSize = Dimensions.HEIGHT
        # 7. Open image path

        img: ImageNdarrayBGR

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

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        with torch.inference_mode():
            # Convert to (bs, c, h, w)
            unsqueezed_tensor = resized_tensor.unsqueeze(0).to(device)

            # predict
            out_bbox: torch.Tensor = model(unsqueezed_tensor)

            # ic(out_bbox)

            xmin, ymin, xmax, ymax = out_bbox[0]
            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))

            starting_point = pt1
            end_point = pt2
            color = (255, 0, 0)
            thickness = 2

            # import bpdb
            # bpdb.set_trace()

            # img = image.astype("uint8")
            # generate the image with bounding box on it
            out_img = cv2.rectangle(
                unsqueezed_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype("uint8"),
                starting_point,
                end_point,
                color,
                thickness,
            )

            # TODO: Enable this?
            # if --display
            # plt.imshow(out_img)

            # NOTE: At this point we have our bounding box for the smaller image, lets figure out what the values would be for a larger image.
            # First setup variables we need
            # -------------------------------------------------------
            image_tensor_to_resize = resized_tensor
            resized_bboxes_tensor = out_bbox[0]
            resized_height = img_height
            resized_width = img_width
            resized_dims = (resized_height, resized_width)

            image_tensor_to_resize.shape[0]
            image_tensor_to_resize.shape[1]
            image_tensor_to_resize.shape[2]

            # perform fullsize transformation
            fullsize_image, fullsize_bboxes = resize_image_and_bbox(
                image_tensor_to_resize,
                resized_bboxes_tensor,
                dims=resized_dims,
                return_percent_coords=False,
                device=device,
            )

            # get fullsize bboxes
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

    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")
    rich.print(f"WOOT Predicted bounding boxes: {fullsize_bboxes}")

    return fullsize_bboxes


# NOTE: pred and store is very important
# NOTE: pred and store is very important
# NOTE: pred and store is very important
# NOTE: pred and store is very important


def resize_image_and_bbox(
    image: torch.Tensor,
    boxes: torch.Tensor,
    dims=(300, 300),
    return_percent_coords=False,
    device: torch.device = None,
):
    """
    Resize image. For the SSD300, resize to (300, 300).
    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.
    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """

    image_tensor_to_resize_height = image.shape[1]
    image_tensor_to_resize_width = image.shape[2]

    # Resize image
    new_image = FT.resize(image, dims)

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

model_name: str = "ScreenCropNetV1_378_epochs.pth"
images_filepath = "/home/pi/screenshot_image_larger00000.JPEG"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# async def load_model(self, model_name: str = "ScreenCropNetV1_378_epochs.pth"):
model = ScreenCropNet_ObjLocModel()
model.name = "ObjLocModelV1"
model.to(device)
weights = f"{aiosettings.screencropnet_dir}/{model_name}"
model = run_get_model_for_inference(model, device, weights)
# self.bot.autocrop_model = model
print(f"Loaded model: {weights} ...")

image, bboxes = predict_from_file(images_filepath, model, device)

# this is the results
# Loaded model: /home/pi/cerebro/ml_models/screencropnet/ScreenCropNetV1_378_epochs.pth ...
# >>>
# >>> image, bboxes = predict_from_file(images_filepath, model, device)
#   0%|                                                                                                                                | 0/1 [00:00<?, ?it/s
# ](224, 224, 3)
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 30.14it/s
# ]
# tensor([[  5.6369, 164.2364, 553.4238, 888.4485]], device='cuda:0')

```


# test goob model

```python
import rich
from goob_ai.utils.imgops import DEVICE, OPENCV_RED, handle_predict_one, predict_from_file, setup_model
from goob_ai import debugger

model = setup_model()
rich.inspect(model, private=True)

path_to_image_from_cli = "/Users/malcolm/dev/bossjones/goob_ai/tests/fixtures/screenshot_image_larger00000.JPEG"

image_results, bboxes_results = predict_from_file(
    path_to_image_from_cli, model, DEVICE
)

image_results
# <PIL.Image.Image image mode=RGB size=560x1214 at 0x33995BBE0>

bboxes_results
# tensor([[ 10.6332, -16.7927,   1.7826, -20.3893]], device='mps:0')
```
