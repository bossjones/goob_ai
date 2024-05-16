"""
This type stub file was generated by pyright.
"""

import av
from typing import Any, Dict, Iterator
from ._video_opt import _HAS_VIDEO_OPT

if _HAS_VIDEO_OPT:
    ...
else:
    ...
if not hasattr(av.video.frame.VideoFrame, "pict_type"):
    av = ...
class VideoReader:
    """
    Fine-grained video-reading API.
    Supports frame-by-frame reading of various streams from a single video
    container. Much like previous video_reader API it supports the following
    backends: video_reader, pyav, and cuda.
    Backends can be set via `torchvision.set_video_backend` function.

    .. betastatus:: VideoReader class

    Example:
        The following examples creates a :mod:`VideoReader` object, seeks into 2s
        point, and returns a single frame::

            import torchvision
            video_path = "path_to_a_test_video"
            reader = torchvision.io.VideoReader(video_path, "video")
            reader.seek(2.0)
            frame = next(reader)

        :mod:`VideoReader` implements the iterable API, which makes it suitable to
        using it in conjunction with :mod:`itertools` for more advanced reading.
        As such, we can use a :mod:`VideoReader` instance inside for loops::

            reader.seek(2)
            for frame in reader:
                frames.append(frame['data'])
            # additionally, `seek` implements a fluent API, so we can do
            for frame in reader.seek(2):
                frames.append(frame['data'])

        With :mod:`itertools`, we can read all frames between 2 and 5 seconds with the
        following code::

            for frame in itertools.takewhile(lambda x: x['pts'] <= 5, reader.seek(2)):
                frames.append(frame['data'])

        and similarly, reading 10 frames after the 2s timestamp can be achieved
        as follows::

            for frame in itertools.islice(reader.seek(2), 10):
                frames.append(frame['data'])

    .. note::

        Each stream descriptor consists of two parts: stream type (e.g. 'video') and
        a unique stream id (which are determined by the video encoding).
        In this way, if the video container contains multiple
        streams of the same type, users can access the one they want.
        If only stream type is passed, the decoder auto-detects first stream of that type.

    Args:
        src (string, bytes object, or tensor): The media source.
            If string-type, it must be a file path supported by FFMPEG.
            If bytes, should be an in-memory representation of a file supported by FFMPEG.
            If Tensor, it is interpreted internally as byte buffer.
            It must be one-dimensional, of type ``torch.uint8``.

        stream (string, optional): descriptor of the required stream, followed by the stream id,
            in the format ``{stream_type}:{stream_id}``. Defaults to ``"video:0"``.
            Currently available options include ``['video', 'audio']``

        num_threads (int, optional): number of threads used by the codec to decode video.
            Default value (0) enables multithreading with codec-dependent heuristic. The performance
            will depend on the version of FFMPEG codecs supported.
    """
    def __init__(self, src: str, stream: str = ..., num_threads: int = ...) -> None:
        ...
    
    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        ...
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ...
    
    def seek(self, time_s: float, keyframes_only: bool = ...) -> VideoReader:
        """Seek within current stream.

        Args:
            time_s (float): seek time in seconds
            keyframes_only (bool): allow to seek only to keyframes

        .. note::
            Current implementation is the so-called precise seek. This
            means following seek, call to :mod:`next()` will return the
            frame with the exact timestamp if it exists or
            the first frame with timestamp larger than ``time_s``.
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Returns video metadata

        Returns:
            (dict): dictionary containing duration and frame rate for every stream
        """
        ...
    
    def set_current_stream(self, stream: str) -> bool:
        """Set current stream.
        Explicitly define the stream we are operating on.

        Args:
            stream (string): descriptor of the required stream. Defaults to ``"video:0"``
                Currently available stream types include ``['video', 'audio']``.
                Each descriptor consists of two parts: stream type (e.g. 'video') and
                a unique stream id (which are determined by video encoding).
                In this way, if the video container contains multiple
                streams of the same type, users can access the one they want.
                If only stream type is passed, the decoder auto-detects first stream
                of that type and returns it.

        Returns:
            (bool): True on success, False otherwise
        """
        ...
    


