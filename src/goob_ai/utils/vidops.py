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
import subprocess
import sys
import tempfile
import time
import traceback
import typing
import uuid

from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Tuple, Union

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

from goob_ai import db, helpers, shell, utils
from goob_ai.shell import _aio_run_process_and_communicate
from goob_ai.utils import file_functions
from goob_ai.utils.devices import get_device
from goob_ai.utils.torchutils import load_model


def calculate_bitrate(duration: float, multiplier: int) -> int:
    """
    Calculate bitrate based on duration and multiplier.

    Args:
        duration (float): The duration of the media file in seconds.
        multiplier (int): A multiplier to adjust the bitrate calculation.

    Returns:
        int: The calculated bitrate in kbps.
    """
    return int(multiplier * 8 * 1000 / duration)


async def process_video(input_file: Path) -> None:
    """
    Process and compress a video file using FFmpeg.

    This function calculates the appropriate bitrate based on the video duration,
    then compresses the video using FFmpeg with the calculated bitrate.

    Args:
        input_file (Path): The path to the input video file.

    Returns:
        None
    """
    LOGGER.debug(f"Processing video file: {input_file}")

    # Calculate bitrate based on input file duration
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_file),
    ]
    duration = float(await _aio_run_process_and_communicate(duration_cmd))
    bitrate = calculate_bitrate(duration, 23)

    LOGGER.debug(f"Video length: {duration}s")
    LOGGER.debug(f"Bitrate target: {bitrate}k")

    # Exit if target bitrate is under 150kbps
    if bitrate < 150:
        LOGGER.debug("Target bitrate is under 150kbps.")
        LOGGER.debug("Unable to compress.")
        return

    video_bitrate = int(bitrate * 90 / 100)
    audio_bitrate = int(bitrate * 10 / 100)

    LOGGER.debug(f"Video Bitrate: {video_bitrate}k")
    LOGGER.debug(f"Audio Bitrate: {audio_bitrate}k")

    # Exit if target video bitrate is under 125kbps
    if video_bitrate < 125:
        LOGGER.debug("Target video bitrate is under 125kbps.")
        LOGGER.debug("Unable to compress.")
        return

    # Exit if target audio bitrate is under 32kbps
    if audio_bitrate < 32:
        LOGGER.debug("Target audio bitrate is under 32.")
        LOGGER.debug("Unable to compress.")
        return

    LOGGER.debug("Compressing video file using FFmpeg...")
    output_file = input_file.parent / f"25MB_{input_file.stem}.mp4"
    compress_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-threads",
        "0",
        "-hwaccel",
        "auto",
        "-i",
        str(input_file),
        "-preset",
        "slow",
        "-c:v",
        "libx264",
        f"-b:v",
        f"{video_bitrate}k",
        "-c:a",
        "aac",
        f"-b:a",
        f"{audio_bitrate}k",
        "-bufsize",
        f"{bitrate}k",
        "-minrate",
        "100k",
        "-maxrate",
        f"{bitrate}k",
        str(output_file),
    ]
    LOGGER.debug(f"compress_cmd = {compress_cmd}")
    await _aio_run_process_and_communicate(compress_cmd)


async def process_audio(input_file: Path) -> None:
    """
    Process and compress an audio file using FFmpeg.

    This function calculates the appropriate bitrate based on the audio duration,
    then compresses the audio using FFmpeg with the calculated bitrate.

    Args:
        input_file (Path): The path to the input audio file.

    Returns:
        None
    """
    LOGGER.debug(f"Processing audio file: {input_file}")

    # Calculate input file duration
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_file),
    ]
    LOGGER.debug(f"duration_cmd = {duration_cmd}")
    duration = float(await _aio_run_process_and_communicate(duration_cmd))
    bitrate = calculate_bitrate(duration, 25)

    LOGGER.debug(f"Audio duration: {duration}s")
    LOGGER.debug(f"Bitrate target: {bitrate}k")

    # Exit if target bitrate is under 32kbps
    if bitrate < 32:
        LOGGER.debug("Target bitrate is under 32kbps.")
        LOGGER.debug("Unable to compress.")
        return

    LOGGER.debug("Compressing audio file using FFmpeg...")
    output_file = input_file.parent / f"25MB_{input_file.stem}.mp3"
    compress_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-stats",
        "-i",
        str(input_file),
        "-preset",
        "slow",
        "-c:a",
        "libmp3lame",
        f"-b:a",
        f"{bitrate}k",
        "-bufsize",
        f"{bitrate}k",
        "-minrate",
        "100k",
        "-maxrate",
        f"{bitrate}k",
        str(output_file),
    ]
    LOGGER.debug(f"compress_cmd = {compress_cmd}")
    await _aio_run_process_and_communicate(compress_cmd)
