"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple, Union
from torch import Tensor, nn

def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    ...

def check_roi_boxes_shape(boxes: Union[Tensor, List[Tensor]]): # -> None:
    ...

def split_normalization_params(model: nn.Module, norm_classes: Optional[List[type]] = ...) -> Tuple[List[Tensor], List[Tensor]]:
    ...

