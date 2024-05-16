"""
This type stub file was generated by pyright.
"""

import pathlib
from typing import Any, Callable, Optional, Tuple, Union
from .vision import VisionDataset

class FER2013(VisionDataset):
    """`FER2013
    <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``root/fer2013`` exists.
        split (string, optional): The dataset split, supports ``"train"`` (default), or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """
    _RESOURCES = ...
    def __init__(self, root: Union[str, pathlib.Path], split: str = ..., transform: Optional[Callable] = ..., target_transform: Optional[Callable] = ...) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        ...
    
    def extra_repr(self) -> str:
        ...
    


