"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_poolformer import PoolFormerConfig

""" PyTorch PoolFormer model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
def drop_path(input: torch.Tensor, drop_prob: float = ..., training: bool = ...) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    ...

class PoolFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class PoolFormerEmbeddings(nn.Module):
    """
    Construct Patch Embeddings.
    """
    def __init__(self, hidden_size, num_channels, patch_size, stride, padding, norm_layer=...) -> None:
        ...
    
    def forward(self, pixel_values): # -> Any:
        ...
    


class PoolFormerGroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group. Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs) -> None:
        ...
    


class PoolFormerPooling(nn.Module):
    def __init__(self, pool_size) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class PoolFormerOutput(nn.Module):
    def __init__(self, config, dropout_prob, hidden_size, intermediate_size) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class PoolFormerLayer(nn.Module):
    """This corresponds to the 'PoolFormerBlock' class in the original implementation."""
    def __init__(self, config, num_channels, pool_size, hidden_size, intermediate_size, drop_path) -> None:
        ...
    
    def forward(self, hidden_states): # -> tuple[Any]:
        ...
    


class PoolFormerEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values, output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutputWithNoAttention:
        ...
    


class PoolFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = PoolFormerConfig
    base_model_prefix = ...
    main_input_name = ...


POOLFORMER_START_DOCSTRING = ...
POOLFORMER_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top.", POOLFORMER_START_DOCSTRING)
class PoolFormerModel(PoolFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Any:
        ...
    
    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        ...
    


class PoolFormerFinalPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


@add_start_docstrings("""
    PoolFormer Model transformer with an image classification head on top
    """, POOLFORMER_START_DOCSTRING)
class PoolFormerForImageClassification(PoolFormerPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(POOLFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


