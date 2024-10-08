# pylint: disable=no-member
"""goob_ai.cogs.autocrop"""

from __future__ import annotations

import os.path
import pathlib

from collections import OrderedDict
from os import PathLike

import cv2
import discord
import numpy as np
import torch

from loguru import logger as LOGGER
from torch import nn

from goob_ai.gen_ai.arch.ScreenCropNet import ObjLocModel as ScreenCropNet_ObjLocModel


# NOTE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
def load_model_for_inference(save_path: str | PathLike, device: str) -> nn.Module:
    """
    Summary:
    Load a model for inference from a specified path and device.

    Explanation:
    This function loads a neural network model for inference from the provided path and device. It adjusts the state dictionary keys, loads the model parameters, and returns the loaded model for inference.
    """
    # LOGGER.info(f"model = {model}")
    LOGGER.info(f"device = {device}")
    LOGGER.info(f"type(device) = {type(device)}")
    LOGGER.info(f"save_path = {save_path}")

    model = ScreenCropNet_ObjLocModel()
    model.name = "ObjLocModelV1"
    state_dict = torch.load(save_path, map_location=device)

    # SOURCE: https://github.com/billhhh/MnasNet-pytorch-pretrained/issues/1
    # fix state_dict key error
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict, strict=False)
    # DISABLED:
    # model.load_state_dict(new_state_dict)

    # model.load_state_dict()
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
    path_to_model: pathlib.PosixPath | PathLike | str,
) -> torch.nn.Module:
    """
    Wrapper function to load model .pth file from disk

    Args:
    ----
        model (torch.nn.Module): _description_
        device (torch.device): _description_
        class_names (List[str]): _description_

    Returns:
    -------
        Tuple[pathlib.PosixPath, torch.nn.Module]: _description_

    """
    LOGGER.info(f"model = {model}")
    LOGGER.info(f"device = {device}")
    LOGGER.info(f"path_to_model = {path_to_model}")

    return load_model_for_inference(path_to_model, device)


# async def load_model(device: torch.device, model_name: str = "ScreenCropNetV1_378_epochs.pth"):
def load_model(device: torch.device, model_name: str = "ScreenNetV1.pth"):
    """
    Summary:
    Load a model for inference on a specified device.

    Explanation:
    This asynchronous function loads a model for inference on the provided device. It initializes the model, moves it to the specified device, loads weights, runs the model for inference, and returns the loaded model for further processing.
    """
    model = ScreenCropNet_ObjLocModel()
    model.name = "ObjLocModelV1"

    # TODO: wait on enabling this till we can figure out why everything is so different
    # # toggle to use eager/graph execution mode
    # model = torch.compile(model)

    # SOURCE: https://towardsdatascience.com/tips-and-tricks-for-upgrading-to-pytorch-2-3127db1d1f3d
    # replace with this to verfiy that error is not in TorchDynamo
    # model = torch.compile(model, 'eager')
    # replace with this to verfiy that error is not in AOTAutograd
    # model = torch.compile(model, 'aot_eager')

    model.to(device)
    # NOTE: Temporary
    my_custom_ml_models_path = "/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/"
    weights = f"{my_custom_ml_models_path}{model_name}"
    model = run_get_model_for_inference(model, device, weights)
    autocrop_model = model
    LOGGER.info(f"Loaded model: {weights} ...")
    return autocrop_model
