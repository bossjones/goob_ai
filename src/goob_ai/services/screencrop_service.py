# pylint: disable=no-member
# pylint: disable=consider-using-from-import
# SOURCE: https://github.com/Kav-K/GPTDiscord/blob/main/services/image_service.py
from __future__ import annotations

import asyncio
import pathlib
import random
import tempfile
import traceback

from io import BytesIO
from os import PathLike
from typing import Dict, List, Literal, Tuple
from urllib.parse import urlparse

import aiohttp
import cv2
import discord
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision

from loguru import logger as LOGGER
from PIL import Image

from goob_ai.utils import async_
from goob_ai.utils.imgops import (
    DEVICE,
    OPENCV_RED,
    display_normalized_rectangle,
    handle_predict_one,
    normalize_rectangle_coords,
    predict_from_file,
    setup_model,
)


def display_image_grid(
    images_filepaths: List[str] | List[PathLike],
    cols=5,
    model: torch.nn.Module | None = None,
    device: torch.device = DEVICE,
):
    LOGGER.info(f"images_filepaths: {images_filepaths}")

    rows = len(images_filepaths) // cols
    # figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    # import bpdb

    # bpdb.set_trace()
    figure: matplotlib.figure.Figure
    ax: np.ndarray
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(30, 10))
    for i, image_filepath in enumerate(images_filepaths):
        image, bboxes = predict_from_file(image_filepath, model, device)

        # set_trace()

        img_as_array = np.asarray(image)

        # SOURCE: https://github.com/opencv/opencv/issues/24522
        # SOURCE: https://github.com/intel/cloud-native-ai-pipeline/pull/179/files
        frame: np.ndarray = img_as_array.copy()

        xmin_fullsize: torch.Tensor
        ymin_fullsize: torch.Tensor
        xmax_fullsize: torch.Tensor
        ymax_fullsize: torch.Tensor

        # get fullsize bboxes
        xmin_fullsize, ymin_fullsize, xmax_fullsize, ymax_fullsize = bboxes[0]

        pt1_fullsize: Tuple[int, int] = (int(xmin_fullsize), int(ymin_fullsize))
        pt2_fullsize: Tuple[int, int] = (int(xmax_fullsize), int(ymax_fullsize))

        starting_point_fullsize: Tuple[int, int] = pt1_fullsize
        end_point_fullsize: Tuple[int, int] = pt2_fullsize
        color: Tuple[Literal[255] | Literal[0], Literal[0]] = OPENCV_RED
        thickness: int = 2

        out_img = cv2.rectangle(
            frame,
            starting_point_fullsize,
            end_point_fullsize,
            color,
            thickness,
        )
        ax.ravel()[i].imshow(out_img)
        # import bpdb

        # bpdb.set_trace()
        # ax.ravel()[i].set_title(bboxes, color="green")
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_image_with_predicted_label(
    to_disk: bool = True,
    img: Image = None,
    target_image_pred_label: torch.Tensor = None,
    target_image_pred_probs: torch.Tensor = None,
    class_names: List[str] = None,
    fname: str = "plot.png",
):
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)

    if to_disk:
        plt.imsave(fname, img)


# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str | PathLike,
    class_names: List[str],
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,  # pyright: ignore[reportAttributeAccessIssue]
    device: torch.device = DEVICE,
    y_preds: List[torch.Tensor] = [],  # pyright: ignore[reportCallInDefaultInitializer]
    y_pred_tensor: torch.Tensor = None,
):
    # 2. Open image
    img = Image.open(image_path)

    ## Predict on image ###
    # 8. Transform the image, add batch dimension and put image on target device
    transformed_image = transform(img).unsqueeze(dim=0)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])

        # 7. Make a prediction on image with an extra dimension and send it to the target device
        # breakpoint()
        # NOTE: Try running the line below manually, it needs to be on same device.
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # boss: Put predictions on CPU for evaluation
    # source: https://www.learnpytorch.io/03_pytorch_computer_vision/#11-save-and-load-best-performing-model
    # ic(target_image_pred_probs)
    y_preds.append(target_image_pred_probs.cpu())
    # boss: Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    image_path_api = pathlib.Path(image_path).resolve()
    plot_fname = f"prediction-{model.name}-{image_path_api.stem}.png"

    # 10. Plot image with predicted label and probability
    plot_image_with_predicted_label(
        to_disk=True,
        img=img,
        target_image_pred_label=target_image_pred_label,
        target_image_pred_probs=target_image_pred_probs,
        class_names=class_names,
        fname=plot_fname,
    )


class ImageService:
    """
    Summary:
    Initialize the ScreenCrop service.

    Explanation:
    This class initializes the ScreenCrop service, which is responsible for processing screen cropping operations.
    """

    def __init__(self):
        # self.screen_crop_model = setup_model()
        pass

    @async_.to_async
    @staticmethod
    def bindingbox_handler(
        img_fpaths: str,
        cols: int = 5,
        model: torch.nn.Module | None = None,
        device: torch.device = DEVICE,
        resize: bool = False,
    ) -> List[Dict]:
        """
        Summary:
        Handle bounding box predictions for images.

        Explanation:
        This static method processes image file paths to predict bounding boxes using a specified model and device. It returns the bounding box results for further processing.
        """

        if model is None:
            model = setup_model()

        image_results, bboxes_results = handle_predict_one(
            img_fpaths, cols=5, model=model, device=device, resize=resize
        )
        LOGGER.info(f"image_results: {image_results}")
        LOGGER.info(f"bboxes_results: {bboxes_results}")
        return bboxes_results

    @staticmethod
    def download_and_predict(
        url: str,
        data_path: pathlib.PosixPath,
        class_names: List[str],
        model: torch.nn.Module | None = None,
        device: torch.device = None,
    ):
        # Download custom image
        urlparse(url).path
        fname = pathlib.Path(urlparse(url).path).name

        # Setup custom image path
        custom_image_path = data_path / fname

        print(f"fname: {custom_image_path}")

        # Download the image if it doesn't already exist
        if not custom_image_path.is_file():
            with open(custom_image_path, "wb") as f:
                # When downloading from GitHub, need to use the "raw" file link
                request = requests.get(url)
                print(f"Downloading {custom_image_path}...")
                f.write(request.content)
        else:
            print(f"{custom_image_path} already exists, skipping download.")

        # Predict on custom image
        pred_and_plot_image(
            model=model,
            image_path=custom_image_path,
            class_names=class_names,
            device=device,
        )

    # @async_.to_async
    @staticmethod
    def handle_predict_from_file(
        path_to_image_from_cli: str | PathLike,
        model: torch.nn.Module | None = None,
        device: torch.device = DEVICE,
    ):
        if model is None:
            model = setup_model()

        predict_from_file(
            path_to_image_from_cli,
            model,
            device,
        )

    # @async_.to_async
    @staticmethod
    def handle_predict_and_display(
        path_to_image_from_cli: List[str] | List[PathLike],
        model: torch.nn.Module | None = None,
        device: torch.device = DEVICE,
    ):
        if model is None:
            model = setup_model()

        display_image_grid(
            path_to_image_from_cli, cols=5, model=model, device=device
        )  # mypy: disable-error-code="arg-type"

    # @async_.to_async
    @staticmethod
    def handle_final(
        path_to_image_from_cli: List[str] | List[PathLike],
        model: torch.nn.Module | None = None,
        device: torch.device = DEVICE,
    ):
        if model is None:
            model = setup_model()

        image_results, bboxes_results = predict_from_file(path_to_image_from_cli, model, DEVICE)

        # FIXME: don't do this, explicity call args in func
        image_and_bbox = [image_results, bboxes_results]

        # normalized creds
        out_bbox_norm = normalize_rectangle_coords(image_and_bbox)

        image_data_with_bounding_box = display_normalized_rectangle(image_results, out_bbox_norm)

        plt.imshow(image_data_with_bounding_box)
