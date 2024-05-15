"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_torch_available, is_vision_available
from .base import Pipeline, build_pipeline_init_args

if is_vision_available():
    ...
if is_torch_available():
    ...
logger = ...
Prediction = Dict[str, Any]
Predictions = List[Prediction]
@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> segmenter = pipeline(model="facebook/detr-resnet-50-panoptic")
    >>> segments = segmenter("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    >>> len(segments)
    2

    >>> segments[0]["label"]
    'bird'

    >>> segments[1]["label"]
    'bird'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the bird on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (768, 512)
    ```


    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def __call__(self, images, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                Mask overlap threshold to eliminate small, disconnected segments.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            list of dictionaries, if the input is a list of several images, will return a list of list of dictionaries
            corresponding to each image.

            The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
            the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
              the original image. Returns a mask filled with zeros if no object is found.
            - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
              "object" described by the label and the mask.
        """
        ...
    
    def preprocess(self, image, subtask=..., timeout=...): # -> BatchFeature:
        ...
    
    def postprocess(self, model_outputs, subtask=..., threshold=..., mask_threshold=..., overlap_mask_area_threshold=...): # -> list[Any]:
        ...
    


