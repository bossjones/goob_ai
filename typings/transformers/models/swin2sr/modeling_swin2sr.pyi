"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_swin2sr import Swin2SRConfig

""" PyTorch Swin2SR Transformer model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
@dataclass
class Swin2SREncoderOutput(ModelOutput):
    """
    Swin2SR encoder's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: torch.FloatTensor = ...
    hidden_states: Optional[Tuple[torch.FloatTensor]] = ...
    attentions: Optional[Tuple[torch.FloatTensor]] = ...


def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    ...

def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    ...

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

class Swin2SRDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    
    def extra_repr(self) -> str:
        ...
    


class Swin2SREmbeddings(nn.Module):
    """
    Construct the patch and optional position embeddings.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        ...
    


class Swin2SRPatchEmbeddings(nn.Module):
    def __init__(self, config, normalize_patches=...) -> None:
        ...
    
    def forward(self, embeddings: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        ...
    


class Swin2SRPatchUnEmbeddings(nn.Module):
    r"""Image to Patch Unembedding"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, embeddings, x_size):
        ...
    


class Swin2SRPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = ...) -> None:
        ...
    
    def maybe_pad(self, input_feature, height, width): # -> Tensor:
        ...
    
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        ...
    


class Swin2SRSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=...) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class Swin2SRSelfOutput(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class Swin2SRAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class Swin2SRIntermediate(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Swin2SROutput(nn.Module):
    def __init__(self, config, dim) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Swin2SRLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=..., pretrained_window_size=...) -> None:
        ...
    
    def get_attn_mask(self, height, width, dtype): # -> Tensor | None:
        ...
    
    def maybe_pad(self, hidden_states, height, width): # -> tuple[Tensor, tuple[Literal[0], Literal[0], Literal[0], Any, Literal[0], Any]]:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
    


class Swin2SRStage(nn.Module):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class Swin2SREncoder(nn.Module):
    def __init__(self, config, grid_size) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, Swin2SREncoderOutput]:
        ...
    


class Swin2SRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Swin2SRConfig
    base_model_prefix = ...
    main_input_name = ...
    supports_gradient_checkpointing = ...


SWIN2SR_START_DOCSTRING = ...
SWIN2SR_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.", SWIN2SR_START_DOCSTRING)
class Swin2SRModel(Swin2SRPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Swin2SRPatchEmbeddings:
        ...
    
    def pad_and_normalize(self, pixel_values):
        ...
    
    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: torch.FloatTensor, head_mask: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        ...
    


class Upsample(nn.Module):
    """Upsample module.

    Args:
        scale (`int`):
            Scale factor. Supported scales: 2^n and 3.
        num_features (`int`):
            Channel number of intermediate features.
    """
    def __init__(self, scale, num_features) -> None:
        ...
    
    def forward(self, hidden_state): # -> Any:
        ...
    


class UpsampleOneStep(nn.Module):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)

    Used in lightweight SR to save parameters.

    Args:
        scale (int):
            Scale factor. Supported scales: 2^n and 3.
        in_channels (int):
            Channel number of intermediate features.
        out_channels (int):
            Channel number of output features.
    """
    def __init__(self, scale, in_channels, out_channels) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, config, num_features) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class NearestConvUpsampler(nn.Module):
    def __init__(self, config, num_features) -> None:
        ...
    
    def forward(self, sequence_output): # -> Any:
        ...
    


class PixelShuffleAuxUpsampler(nn.Module):
    def __init__(self, config, num_features) -> None:
        ...
    
    def forward(self, sequence_output, bicubic, height, width): # -> tuple[Any, Any]:
        ...
    


@add_start_docstrings("""
    Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.
    """, SWIN2SR_START_DOCSTRING)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageSuperResolutionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, ImageSuperResolutionOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> import numpy as np
         >>> from PIL import Image
         >>> import requests

         >>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

         >>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
         >>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

         >>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
         >>> image = Image.open(requests.get(url, stream=True).raw)
         >>> # prepare image for the model
         >>> inputs = processor(image, return_tensors="pt")

         >>> # forward pass
         >>> with torch.no_grad():
         ...     outputs = model(**inputs)

         >>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
         >>> output = np.moveaxis(output, source=0, destination=-1)
         >>> output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
         >>> # you can visualize `output` with `Image.fromarray`
         ```"""
        ...
    


