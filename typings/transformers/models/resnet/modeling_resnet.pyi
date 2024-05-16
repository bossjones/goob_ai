"""
This type stub file was generated by pyright.
"""

import torch
from typing import Optional
from torch import Tensor, nn
from ...modeling_outputs import BackboneOutput, BaseModelOutputWithNoAttention, BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import BackboneMixin
from .configuration_resnet import ResNetConfig

""" PyTorch ResNet model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
_EXPECTED_OUTPUT_SHAPE = ...
_IMAGE_CLASS_CHECKPOINT = ...
_IMAGE_CLASS_EXPECTED_OUTPUT = ...
class ResNetConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = ..., stride: int = ..., activation: str = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class ResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """
    def __init__(self, config: ResNetConfig) -> None:
        ...
    
    def forward(self, pixel_values: Tensor) -> Tensor:
        ...
    


class ResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class ResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = ..., activation: str = ...) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class ResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = ..., activation: str = ..., reduction: int = ..., downsample_in_bottleneck: bool = ...) -> None:
        ...
    
    def forward(self, hidden_state):
        ...
    


class ResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """
    def __init__(self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = ..., depth: int = ...) -> None:
        ...
    
    def forward(self, input: Tensor) -> Tensor:
        ...
    


class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        ...
    
    def forward(self, hidden_state: Tensor, output_hidden_states: bool = ..., return_dict: bool = ...) -> BaseModelOutputWithNoAttention:
        ...
    


class ResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ResNetConfig
    base_model_prefix = ...
    main_input_name = ...


RESNET_START_DOCSTRING = ...
RESNET_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare ResNet model outputting raw features without any specific head on top.", RESNET_START_DOCSTRING)
class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_CHECKPOINT_FOR_DOC, output_type=BaseModelOutputWithPoolingAndNoAttention, config_class=_CONFIG_FOR_DOC, modality="vision", expected_output=_EXPECTED_OUTPUT_SHAPE)
    def forward(self, pixel_values: Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BaseModelOutputWithPoolingAndNoAttention:
        ...
    


@add_start_docstrings("""
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """, RESNET_START_DOCSTRING)
class ResNetForImageClassification(ResNetPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(checkpoint=_IMAGE_CLASS_CHECKPOINT, output_type=ImageClassifierOutputWithNoAttention, config_class=_CONFIG_FOR_DOC, expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    ResNet backbone, to be used with frameworks like DETR and MaskFormer.
    """, RESNET_START_DOCSTRING)
class ResNetBackbone(ResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```"""
        ...
    


