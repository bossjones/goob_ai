"""
This type stub file was generated by pyright.
"""

import torch
import av
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

if not hasattr(av.video.frame.VideoFrame, "pict_type"):
    av = ...
_CALLED_TIMES = ...
_GC_COLLECTION_INTERVAL = ...
def write_video(filename: str, video_array: torch.Tensor, fps: float, video_codec: str = ..., options: Optional[Dict[str, Any]] = ..., audio_array: Optional[torch.Tensor] = ..., audio_fps: Optional[float] = ..., audio_codec: Optional[str] = ..., audio_options: Optional[Dict[str, Any]] = ...) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    ...

def read_video(filename: str, start_pts: Union[float, Fraction] = ..., end_pts: Optional[Union[float, Fraction]] = ..., pts_unit: str = ..., output_format: str = ...) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    ...

def read_video_timestamps(filename: str, pts_unit: str = ...) -> Tuple[List[int], Optional[float]]:
    """
    List the video frames timestamps.

    Note that the function decodes the whole video frame-by-frame.

    Args:
        filename (str): path to the video file
        pts_unit (str, optional): unit in which timestamp values will be returned
            either 'pts' or 'sec'. Defaults to 'pts'.

    Returns:
        pts (List[int] if pts_unit = 'pts', List[Fraction] if pts_unit = 'sec'):
            presentation timestamps for each one of the frames in the video.
        video_fps (float, optional): the frame rate for the video

    """
    ...

