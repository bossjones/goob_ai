# SOURCE: https://github.com/Kav-K/GPTDiscord/blob/main/services/image_service.py
from __future__ import annotations

import asyncio
import random
import tempfile
import traceback

from io import BytesIO
from typing import Dict, List

import aiohttp
import discord
import torch

from loguru import logger as LOGGER
from PIL import Image

from goob_ai.utils import async_
from goob_ai.utils.imgops import DEVICE, handle_predict_one, setup_model


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
