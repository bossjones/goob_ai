# pylint: disable=consider-using-with, consider-using-min-builtin
# SOURCE: https://github.com/GDi4K/imitools/blob/main/imitools.py
from __future__ import annotations

import io
import math
import os
import tempfile

from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import PathLike
from pathlib import Path
from typing import Any, Callable, List, Union

import requests
import torch

from IPython.core.display import HTML
from loguru import logger as LOGGER
from matplotlib import pyplot as plt
from PIL import Image, UnidentifiedImageError
from pkg_resources import parse_version
from torchvision.transforms import transforms


_last_search_wrapper = None
N_WORKERS = min(10, os.cpu_count())


class ImageDefaults:
    """
    A class to hold default settings for image processing.

    Attributes:
        device (str): The device to be used for image processing, default is "cpu".
    """

    def __init__(self) -> None:
        """Initialize the ImageDefaults class with default settings."""
        self.device = "cpu"


defaults = ImageDefaults()


def download_image(img_url: str) -> Image.Image | None:
    """
    Download an image from a given URL.

    This function downloads an image from the specified URL and returns it as a PIL Image object.
    If the download fails or the image cannot be opened, it returns None.

    Args:
        img_url (str): The URL of the image to be downloaded.

    Returns:
        Image.Image | None: The downloaded image as a PIL Image object, or None if the download fails.
    """
    image = None
    try:
        buffer = tempfile.SpooledTemporaryFile(max_size=1e9)  # type: ignore
        r = requests.get(img_url, stream=True)
        if r.status_code == 200:
            for chunk in r.iter_content(chunk_size=1024):
                buffer.write(chunk)
            buffer.seek(0)
            image = Image.open(io.BytesIO(buffer.read()))
        buffer.close()
        return image
    except Exception:
        return image


# based on https://gist.github.com/sigilioso/2957026
def image_crop(img: Image.Image, size: tuple[int, int], crop_type: str = "middle") -> Image.Image:
    """
    Crop an image to the specified size.

    This function resizes and crops an image to the desired size. The cropping can be done
    from the top, middle, or bottom of the image based on the specified crop type.

    Args:
        img (Image.Image): The input image to be cropped.
        size (tuple[int, int]): The desired size (width, height) of the cropped image.
        crop_type (str, optional): The type of cropping to perform. Can be 'top', 'middle', or 'bottom'. Defaults to 'middle'.

    Returns:
        Image.Image: The cropped image.

    Raises:
        ValueError: If an invalid value is provided for crop_type.
    """
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], round(size[0] * img.size[1] / img.size[0])), Image.Resampling.BILINEAR)
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, img.size[0], size[1])
        elif crop_type == "middle":
            box = (0, round((img.size[1] - size[1]) / 2), img.size[0], round((img.size[1] + size[1]) / 2))
        elif crop_type == "bottom":
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            """
            Display images in multiple rows.

            If there are multiple rows of images, they are displayed in a grid with optional captions.
            """
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((round(size[1] * img.size[0] / img.size[1]), size[1]), Image.Resampling.BILINEAR)
        # Crop in the top, middle or bottom
        if crop_type == "top":
            box = (0, 0, size[0], img.size[1])
        elif crop_type == "middle":
            box = (round((img.size[0] - size[0]) / 2), 0, round((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == "bottom":
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError("ERROR: invalid value for crop_type")
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]), Image.Resampling.BILINEAR)

    return img


def thread_loop(fn: Callable[[Any], Any], input_array: List[Any], n_workers: int = N_WORKERS) -> List[Any]:
    return_data = []

    with ThreadPoolExecutor(n_workers) as executor:
        futures = [executor.submit(fn, input_item) for input_item in input_array]

        for future in as_completed(futures):
            result = future.result()
            return_data.append(result)

    return return_data


class VideoWrapper:
    """
    A class to wrap video file information.

    Attributes:
        video_path (str): The path to the video file.
        video_size (tuple[int, int]): The size of the video (width, height).
    """

    def __init__(self, video_path: str, video_size: tuple[int, int]) -> None:
        """
        Initialize the VideoWrapper class.

        Args:
            video_path (str): The path to the video file.
            video_size (tuple[int, int]): The size of the video (width, height).
        """
        self.video_path = video_path
        self.video_size = video_size

    def path(self) -> str:
        """
        Get the path to the video file.

        Returns:
            str: The path to the video file.
        """
        return self.video_path

    # def show(self) -> HTML:
    #     mp4 = open(self.video_path, "rb").read()
    #     data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    #     width, height = self.video_size

    #     return HTML(
    #         f"""
    #     <video width={width} height={height} controls>
    #           <source src="%s" type="video/mp4">
    #     </video>
    #     """
    #         % data_url
    #     )


class ImageWrapper:
    def __init__(
        self, data: list[Image.Image] | torch.Tensor, image_type: str, labels: list[int] | None = None
    ) -> None:
        """Initialize the ImageWrapper class.

        Args:
            data (list[Image.Image] | torch.Tensor): The image data.
            image_type (str): The type of the image data ('pil' or 'pt').
            labels (list[int] | None, optional): The labels for the images. Defaults to None.
        """
        self.data: list[Image.Image] | torch.Tensor = data
        self.image_type: str = image_type
        self.labels: list[int] = list(range(len(data))) if labels is None else labels

    def resize(self, size: tuple[int, int] = (256, 256), **kwargs: Any) -> ImageWrapper:
        """Resize the images to the specified size.

        Args:
            size (tuple[int, int], optional): The desired size (width, height) of the resized images. Defaults to (256, 256).
            **kwargs (Any): Additional keyword arguments to be passed to the resize method of the PIL Image class.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the resized images.
        """
        ref = self
        if self.image_type != "pil":
            ref = self.cpil()

        if not isinstance(size, tuple):
            size = (size, size)

        i_size = (int(size[0]), int(size[1]))

        new_images = [im.resize(i_size, **kwargs) for im in ref.data]
        return ImageWrapper(new_images, "pil")

    def crop(self, size: tuple[int, int] = (256, 256), crop_type: str = "middle") -> ImageWrapper:
        """Crop the images to the specified size.

        Args:
            size (tuple[int, int], optional): The desired size (width, height) of the cropped images. Defaults to (256, 256).
            crop_type (str, optional): The type of cropping to perform. Can be 'top', 'middle', or 'bottom'. Defaults to 'middle'.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the cropped images.
        """
        ref = self
        if ref.image_type != "pil":
            ref = ref.cpil()

        if not isinstance(size, tuple):
            size = (size, size)

        i_size = (int(size[0]), int(size[1]))

        new_images = [image_crop(im, size, crop_type) for im in ref.data]
        return ImageWrapper(new_images, "pil")

    def normalize(self) -> ImageWrapper:
        """Normalize the image data.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the normalized image data.
        """
        ref: ImageWrapper = self
        if self.image_type != "pt":
            ref = self.cpt()

        normalized = (ref.data - ref.data.min()) / (ref.data.max() - ref.data.min())
        return ImageWrapper(normalized, "pt")

    def pick(self, *args: int | list[int]) -> ImageWrapper:  # type: ignore
        """Select specific images from the ImageWrapper instance.

        Args:
            *args (int | list[int]): The indexes of the images to be picked. Can be individual integers or a list of integers.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the selected images.

        Raises:
            Exception: If no indexes are provided.
        """
        if not args:
            raise Exception("provide some indexes to pick")

        indexes = list(args)
        if isinstance(args[0], list):
            indexes = args[0]

        if self.image_type == "pil":
            return ImageWrapper([self.data[i] for i in indexes], "pil")

        if self.image_type == "pt":
            return ImageWrapper(self.data[indexes], "pt")

    def sinrange(self: ImageWrapper) -> ImageWrapper:
        """Scale image data to the range [-1, 1].

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the scaled image data.
        """
        ref = self
        if self.image_type != "pt":
            ref = self.cpt()

        return ImageWrapper(ref.data * 2 - 1, "pt")

    def pil(self) -> Image.Image | list[Image.Image]:  # type: ignore
        """Convert the image data to PIL format.

        Returns:
            Image.Image | list[Image.Image]: The image data in PIL format. If there is only one image, it returns a single PIL Image object. Otherwise, it returns a list of PIL Image objects.
        """
        if self.image_type == "pil":
            return self.data[0] if len(self.data) == 1 else self.data

        if self.image_type == "pt":
            make_pil = transforms.ToPILImage()
            pt_images = self.data.cpu()  # type: ignore
            pil_images = [make_pil(i) for i in pt_images]
            return pil_images[0] if len(pil_images) == 1 else pil_images

    def pt(self) -> torch.Tensor:
        """Convert the image data to PyTorch tensor format.

        Returns:
            torch.Tensor: The image data in PyTorch tensor format.
        """
        if self.image_type == "pil":
            pt_images = [transforms.ToTensor()(im) for im in self.data]
            return torch.stack(pt_images).to(defaults.device)

        if self.image_type == "pt":
            return self.data

    def to(self, device: str = "cpu") -> ImageWrapper:
        """Move the image data to the specified device.

        Args:
            device (str, optional): The device to move the image data to. Defaults to "cpu".

        Returns:
            ImageWrapper: A new ImageWrapper instance with the image data moved to the specified device.

        Raises:
            Exception: If the image data is not in PyTorch tensor format.
        """
        if self.image_type != "pt":
            raise Exception("to() only applied for pytorch tensors")

        return ImageWrapper(self.data.to(device), "pt")  # type: ignore

    def cpil(self) -> ImageWrapper:
        """Convert the image data to PIL format.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the image data in PIL format.
        """
        images: list[Image.Image] | Image.Image = self.pil()
        if isinstance(images, Image.Image):
            images = [images]

        return ImageWrapper(images, "pil")

    def cpt(self) -> ImageWrapper:
        """Convert the image data to PyTorch tensor format.

        Returns:
            ImageWrapper: A new ImageWrapper instance containing the image data in PyTorch tensor format.
        """
        return ImageWrapper(self.pt(), "pt")

    def show(
        self,
        cmap: Any = None,
        figsize: tuple[int, int] | None = None,
        cols: int = 6,
        max_count: int = 36,
        scale: int = -1,
        captions: bool = True,
    ) -> None:
        """Display a grid of images.

        Args:
            cmap (Any, optional): The colormap to use for displaying the images. Defaults to None.
            figsize (tuple[int, int] | None, optional): The size of the figure. Defaults to None.
            cols (int, optional): The number of columns in the grid. Defaults to 6.
            max_count (int, optional): The maximum number of images to display. Defaults to 36.
            scale (int, optional): The scale of the images. Defaults to -1.
            captions (bool, optional): Whether to display captions for the images. Defaults to True.
        """
        if len(self.data) == 1:
            """
            Display a single image.

            If there is only one image in the ImageWrapper instance, it is displayed with the specified scale.
            """
            scale = 4 if scale == -1 else scale
            plt.figure(figsize=(scale, scale))
            plt.axis("off")
            if self.image_type == "pil":
                plt.imshow(self.data[0], cmap=cmap)
            else:
                plt.imshow(self.data[0].permute(1, 2, 0).cpu(), cmap=cmap)  # type: ignore

            return

        scale = 2.5 if scale == -1 else scale
        images = self.data.cpu() if self.image_type == "pt" else self.data  # type: ignore
        labels = self.labels
        image_count = len(self.data)

        if image_count > max_count:
            print(
                f"Only showing {max_count} images of the total {image_count}. Use the `max_count` parameter to change it."
            )
            images = self.data[:max_count]
            image_count = max_count

        if image_count < cols:
            cols = image_count

        rows = math.ceil(image_count / cols)

        if figsize is None:
            figsize = figsize = (cols * scale, rows * scale)

        _, ax = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            for i in range(image_count):
                image = images[i] if self.image_type == "pil" else images[i].permute(1, 2, 0)  # type: ignore
                ax[i].imshow(image, cmap=cmap)
                ax[i].axis("off")
                if captions:
                    ax[i].set_title(f"{labels[i]}")
        else:
            for row in range(rows):
                for col in range(cols):
                    i = row * cols + col
                    if i < image_count:
                        image = images[i] if self.image_type == "pil" else images[i].permute(1, 2, 0)  # type: ignore
                        ax[row][col].imshow(image, cmap=cmap)
                        ax[row][col].axis("off")
                        if captions:
                            ax[row][col].set_title(f"{labels[i]}")
                    else:
                        ax[row][col].axis("off")

    def to_dir(self, output_dir: str, prefix: str = "image", max_workers: int = N_WORKERS) -> None:
        """Save images to a specified directory.

        Args:
            output_dir (str): The path to the directory where the images will be saved.
            prefix (str, optional): The prefix for the saved image filenames. Defaults to "image".
            max_workers (int, optional): The maximum number of worker threads to use for saving images. Defaults to N_WORKERS.
        """
        ref = self
        if self.image_type != "pil":
            ref = self.cpil()

        dir_path = Path(output_dir)
        dir_path.mkdir(exist_ok=True, parents=True)

        images = ref.data

        def save_image(i: int) -> None:
            """
            Save an image to the specified directory.

            This function saves an image from the list of images to the specified
            directory with a given prefix and index. If an error occurs during
            the saving process, it prints an error message.

            Args:
                i (int): The index of the image in the list to be saved.

            Raises:
                Exception: If an error occurs during the image saving process.
            """
            try:
                path = Path(output_dir) / f"{prefix}_{i:04}.png"
                images[i].save(path)
            except Exception as e:
                print("image saving error:", e)

        thread_loop(save_image, range(len(images)))  # type: ignore

    def to_video(self, out_path: PathLike | str | None = None, frame_rate: int = 12) -> VideoWrapper:
        """Convert a sequence of images to a video.

        Args:
            out_path (PathLike | str | None, optional): The path to save the video file. If None, a temporary path is used. Defaults to None.
            frame_rate (int, optional): The frame rate of the video. Defaults to 12.

        Returns:
            VideoWrapper: An instance of the VideoWrapper class containing the video file path and size.
        """
        ref = self
        if self.image_type == "pt":
            ref = self.cpil()

        id = int(torch.rand(1)[0].item() * 9999999)
        image_dir = Path(f"/tmp/{id}/images")
        image_dir.mkdir(exist_ok=True, parents=True)

        if out_path is None:
            out_path = f"/tmp/{id}/video.mp4"

        video_path = Path(out_path)  # type: ignore
        video_size = ref.data[0].size
        images_selector = image_dir / "image_%04d.png"

        ref.to_dir(image_dir, prefix="image")  # type: ignore

        command = f"ffmpeg -v 0 -y -f image2 -framerate {frame_rate} -i {images_selector} -c:v h264_nvenc -preset slow -qp 18 -pix_fmt yuv420p {video_path}"
        os.system(command)

        return VideoWrapper(video_path, video_size)  # type: ignore


def wrap(
    input_data: Union[ImageWrapper, torch.Tensor, Image.Image, list[Union[torch.Tensor, Image.Image, ImageWrapper]]],
    labels: list[int] | None = None,
) -> ImageWrapper:
    """
    Wrap various types of image data into an ImageWrapper instance.

    This function takes different types of image data, such as ImageWrapper instances,
    PyTorch tensors, PIL Images, or lists of these types, and wraps them into a single
    ImageWrapper instance.

    Args:
        input_data (Union[ImageWrapper, torch.Tensor, Image.Image, list[Union[torch.Tensor, Image.Image, ImageWrapper]]]):
            The image data to be wrapped.
        labels (list[int] | None, optional): The labels for the images. Defaults to None.

    Returns:
        ImageWrapper: An ImageWrapper instance containing the wrapped image data.

    Raises:
        Exception: If the input data type is not supported.
    """
    if isinstance(input_data, ImageWrapper):
        # If the input data is already an ImageWrapper, return it as is.
        return input_data

    if isinstance(input_data, torch.Tensor):
        # If the input data is a PyTorch tensor, adjust its dimensions and wrap it.
        if len(input_data.shape) == 2:
            input_data = input_data.unsqueeze(0).unsqueeze(0)

        if len(input_data.shape) == 3:
            input_data = input_data.unsqueeze(0)

        return ImageWrapper(input_data.detach().float(), "pt", labels)

    if isinstance(input_data, Image.Image):
        # If the input data is a single PIL Image, wrap it in a list and then wrap it.
        return ImageWrapper([input_data], "pil", labels)

    if isinstance(input_data, list):
        # If the input data is a list, determine the type of its elements and wrap accordingly.
        if isinstance(input_data[0], torch.Tensor):
            images = torch.stack(input_data).squeeze(1).detach().float()
            return ImageWrapper(images, "pt", labels)

        if isinstance(input_data[0], Image.Image):
            return ImageWrapper(input_data, "pil", labels)

        if isinstance(input_data[0], ImageWrapper):
            image_list = list(map(lambda w: w.pt(), input_data))  # type: ignore
            images = torch.stack(image_list).squeeze(1).detach().float()
            return ImageWrapper(images, "pt", labels)

    raise Exception("not implemented!")
    # Raise an exception if the input data type is not supported.


def from_dir(dir_path: str) -> ImageWrapper:
    """
    Load images from a directory and return them as an ImageWrapper instance.

    This function iterates through the files in the specified directory,
    reads the images, and converts them to RGB format. The images are then
    wrapped in an ImageWrapper instance and returned.

    Args:
        dir_path (str): The path to the directory containing the images.

    Returns:
        ImageWrapper: An ImageWrapper instance containing the loaded images.

    Raises:
        UnidentifiedImageError: If a file in the directory is not a valid image.
    """
    file_list = [f for f in Path(dir_path).iterdir() if not f.is_dir()]
    image_list = []

    def read_image(f: Path) -> None:
        """Read an image from a file and convert it to RGB format.

        This function attempts to open an image file and convert it to RGB format.
        If the file is not a valid image, it is ignored.

        Args:
            f (Path): The path to the image file.
        """
        try:
            image_list.append(Image.open(f).convert("RGB"))
        except UnidentifiedImageError:
            None

    thread_loop(read_image, file_list)

    return ImageWrapper(image_list, "pil")


def from_path(input_data: Union[str, Path]) -> ImageWrapper:
    """
    Load an image from a file path and return it as an ImageWrapper instance.

    This function reads an image from the specified file path, converts it to RGB format,
    and wraps it in an ImageWrapper instance.

    Args:
        input_data (Union[str, Path]): The path to the image file.

    Returns:
        ImageWrapper: An ImageWrapper instance containing the loaded image.

    Raises:
        UnidentifiedImageError: If the file at the specified path is not a valid image.
    """
    pil_image = Image.open(input_data).convert("RGB")
    return ImageWrapper([pil_image], "pil")


# class LivePlotter:
#     def __init__(self, cols: int = 2, figsize: tuple[int, int] = (15, 4)) -> None:
#         """
#         Initialize the LivePlotter class.

#         Args:
#             cols (int, optional): The number of columns in the plot. Defaults to 2.
#             figsize (tuple[int, int], optional): The size of the figure. Defaults to (15, 4).

#         Attributes:
#             cols (int): The number of columns in the plot.
#             fig (Figure): The matplotlib figure object.
#             out (DisplayHandle): The display handle for updating the plot.
#             subplots (Axes or array of Axes): The subplots in the figure.
#             queue (list): The queue of plot commands.
#         """
#         fig, subplots = plt.subplots(1, cols, figsize=(20, 5))
#         fig.patch.set_facecolor("white")
#         fig.tight_layout()
#         out = display(fig, display_id=True)

#         self.cols = cols
#         self.fig = fig
#         self.out = out
#         self.subplots = subplots

#         self.queue = []

#     def plot(self, subplot_id: int, *args: Any, **kwargs: Any) -> LivePlotter:
#         """
#         Plot data on the specified subplot.

#         This method queues a command to plot data on the specified subplot.
#         The data to be plotted and any additional keyword arguments are passed
#         to the matplotlib plot function.

#         Args:
#             subplot_id (int): The ID of the subplot where the data will be plotted.
#             *args (Any): Positional arguments to be passed to the matplotlib plot function.
#             **kwargs (Any): Keyword arguments to be passed to the matplotlib plot function.

#         Returns:
#             LivePlotter: The LivePlotter instance with the queued command.
#         """
#         self.queue.append(("plot", subplot_id, args, kwargs))
#         return self

#     def title(self, subplot_id: int, title: str) -> LivePlotter:
#         """
#         Set the title of a subplot.

#         This method queues a command to set the title of the specified subplot.

#         Args:
#             subplot_id (int): The ID of the subplot where the title will be set.
#             title (str): The title text to be set for the subplot.

#         Returns:
#             LivePlotter: The LivePlotter instance with the queued command.
#         """
#         self.queue.append(("title", subplot_id, title))
#         return self

#     def imshow(self, subplot_id: int, image: Union[Image.Image, torch.Tensor]) -> LivePlotter:
#         """
#         Display an image in the specified subplot.

#         This method queues a command to display an image in the specified subplot.
#         The image can be a PIL Image or a PyTorch tensor.

#         Args:
#             subplot_id (int): The ID of the subplot where the image will be displayed.
#             image (Union[Image.Image, torch.Tensor]): The image to be displayed.

#         Returns:
#             LivePlotter: The LivePlotter instance with the queued command.
#         """
#         self.queue.append(("imshow", subplot_id, image))
#         return self

#     def update(self) -> None:
#         """
#         Update the live plot with the queued commands.

#         This method processes the queued commands to update the live plot.
#         It clears the subplots, executes the queued commands (imshow, plot, title),
#         and updates the display with the new plot.

#         Returns:
#             None
#         """
#         for col in range(self.cols):
#             if self.cols == 1:
#                 self.subplots.clear()
#             else:
#                 self.subplots[col].clear()

#         for item in self.queue:
#             if item[0] == "imshow":
#                 _, subplot_id, image = item
#                 if self.cols == 1:
#                     self.subplots.imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
#                     self.subplots.axis("off")
#                 else:
#                     self.subplots[subplot_id].imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
#                     self.subplots[subplot_id].axis("off")

#             if item[0] == "plot":
#                 _, subplot_id, args, kwargs = item
#                 self.subplots[subplot_id].plot(*args, **kwargs)
#                 if "label" in kwargs:
#                     self.subplots[subplot_id].legend()

#             if item[0] == "title":
#                 _, subplot_id, title = item
#                 self.subplots[subplot_id].title.set_text(title)

#         self.queue = []
#         self.out.update(self.fig)
#         for col in range(self.cols):
#             if self.cols == 1:
#                 self.subplots.clear()
#             else:
#                 self.subplots[col].clear()

#         for item in self.queue:
#             if item[0] == "imshow":
#                 _, subplot_id, image = item
#                 if self.cols == 1:
#                     self.subplots.imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
#                     self.subplots.axis("off")
#                 else:
#                     self.subplots[subplot_id].imshow(wrap(image).pt().detach().cpu()[0].permute(1, 2, 0))
#                     self.subplots[subplot_id].axis("off")

#             if item[0] == "plot":
#                 _, subplot_id, args, kwargs = item
#                 self.subplots[subplot_id].plot(*args, **kwargs)
#                 if "label" in kwargs:
#                     self.subplots[subplot_id].legend()

#             if item[0] == "title":
#                 _, subplot_id, title = item
#                 self.subplots[subplot_id].title.set_text(title)

#         self.queue = []
#         self.out.update(self.fig)

#     def close(self) -> None:
#         """
#         Close the matplotlib figure.

#         This method closes the figure associated with the LivePlotter instance,
#         freeing up the resources used by the figure.
#         """
#         plt.close()


# def live_plot(*args: Any, **kwargs: Any) -> LivePlotter:
#     """
#     Create a LivePlotter instance for live plotting.

#     This function initializes a LivePlotter instance with the given arguments and keyword arguments.
#     The LivePlotter instance can be used to create and update live plots.

#     Args:
#         *args (Any): Positional arguments to be passed to the LivePlotter constructor.
#         **kwargs (Any): Keyword arguments to be passed to the LivePlotter constructor.

#     Returns:
#         LivePlotter: An instance of the LivePlotter class for live plotting.
#     """
#     return LivePlotter(*args, **kwargs)


def download(image_urls: Union[str, List[str]]) -> ImageWrapper:
    """
    Download images from the given URLs.

    This function takes a single URL or a list of URLs, downloads the images,
    and returns them wrapped in an ImageWrapper instance.

    Args:
        image_urls (Union[str, List[str]]): A single image URL or a list of image URLs.

    Returns:
        ImageWrapper: An ImageWrapper instance containing the downloaded images.

    Raises:
        Exception: If the image URLs are invalid or the images cannot be downloaded.
    """
    if isinstance(image_urls, str):
        """
        Convert a single URL to a list of URLs.

        If the input is a single URL, it is converted to a list containing that URL.
        """
        image_urls = [image_urls]

    result_list = thread_loop(download_image, image_urls)
    """
    Download images using multiple threads.

    The function uses a thread pool to download images concurrently.
    """
    images = []
    """
    Filter out None values from the downloaded images.

    The function iterates through the downloaded images and filters out any None values.
    """
    images.extend(image for image in result_list if image is not None)
    return wrap(images)


# def merge(
#     *args: Union[
#         ImageWrapper, List[ImageWrapper], List[Image.Image], List[torch.Tensor], ImageWrapper, torch.Tensor, Image.Image
#     ],
# ) -> ImageWrapper:
#     """
#     Merge multiple image data sources into a single ImageWrapper instance.

#     This function takes multiple image data sources, which can be ImageWrapper instances,
#     lists of ImageWrapper instances, lists of PIL Images, lists of PyTorch tensors, or
#     individual PIL Images or PyTorch tensors. It merges these sources into a single
#     ImageWrapper instance.

#     Args:
#         *args (Union[ImageWrapper, List[ImageWrapper], List[Image.Image], List[torch.Tensor], ImageWrapper, torch.Tensor, Image.Image]):
#             The image data sources to be merged.

#     Returns:
#         ImageWrapper: An ImageWrapper instance containing the merged image data.

#     Raises:
#         Exception: If the input data type is not supported.
#     """
#     args = list(args)
#     if isinstance(args[0], list) and not isinstance(args[0][0], Image.Image):
#         args = args[0]

#     wrappers = [wrap(i) for i in args]

#     if wrappers[0].image_type == "pil":
#         images = []
#         for w in wrappers:
#             convered = w.cpil()
#             for i in convered.data:
#                 images.append(i)

#         return ImageWrapper(images, "pil")

#     if wrappers[0].image_type == "pt":
#         tensor_list = [w.pt() for w in wrappers]
#         return ImageWrapper(torch.cat(tensor_list, dim=0), "pt")


# def search_images(prompt: str, max_results: int = 10) -> ImageWrapper:
#     """
#     Search for images using a given prompt.

#     This function uses the DuckDuckGo search engine to find images based on the provided prompt.
#     It returns an ImageWrapper instance containing the downloaded images.

#     Args:
#         prompt (str): The search query to find images.
#         max_results (int, optional): The maximum number of images to retrieve. Defaults to 10.

#     Returns:
#         ImageWrapper: An ImageWrapper instance containing the downloaded images.
#     """
#     image_urls = [item["image"] for item in ddg_images(prompt, max_results=max_results)]
#     global _last_search_wrapper
#     _last_search_wrapper = download(image_urls)
#     return _last_search_wrapper


def search_history() -> ImageWrapper | None:
    """
    Retrieve the last searched images.

    This function returns the ImageWrapper instance containing the images
    from the last search performed using the `search_images` function.
    If no search has been performed yet, it returns None.

    Returns:
        ImageWrapper | None: The ImageWrapper instance containing the last searched images,
                                or None if no search has been performed.
    """
    return _last_search_wrapper


# class ImiTools:
#     def __init__(self):
#         self.defaults = defaults

#     def wrap(self, data, labels=None) -> ImageWrapper:
#         return wrap(data, labels)

#     def from_path(self, path) -> ImageWrapper:
#         return from_path(path)

#     def from_dir(self, path) -> ImageWrapper:
#         return from_dir(path)

#     def live_plot(self, *args, **kwargs) -> LivePlotter:
#         return live_plot(*args, **kwargs)

#     def download(self, img_urls) -> ImageWrapper:
#         return download(img_urls)

#     def merge(self, *args) -> ImageWrapper:
#         return merge(*args)

# I = ImiTools()
# I.defaults.device = device


# from duckduckgo_search import ddg_images
# from IPython.display import HTML, display
# if parse_version(pil.__version__)>=parse_version('10.0.0'):
#     Image.Resampling.BILINEAR=Image.LANCZOS
