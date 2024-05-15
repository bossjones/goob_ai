"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig
from ...utils.backbone_utils import BackboneConfigMixin

""" BiT model configuration"""
logger = ...
class BitConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BitModel`]. It is used to instantiate an BiT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the BiT
    [google/bit-50](https://huggingface.co/google/bit-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"preactivation"`):
            The layer to use, it can be either `"preactivation"` or `"bottleneck"`.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        global_padding (`str`, *optional*):
            Padding strategy to use for the convolutional layers. Can be either `"valid"`, `"same"`, or `None`.
        num_groups (`int`, *optional*, defaults to 32):
            Number of groups used for the `BitGroupNormActivation` layers.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate for the stochastic depth.
        embedding_dynamic_padding (`bool`, *optional*, defaults to `False`):
            Whether or not to make use of dynamic padding for the embedding layer.
        output_stride (`int`, *optional*, defaults to 32):
            The output stride of the model.
        width_factor (`int`, *optional*, defaults to 1):
            The width factor for the model.
        out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). If unset and `out_indices` is set, will default to the
            corresponding stages. If unset and `out_indices` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). If unset and `out_features` is set, will default to the corresponding stages.
            If unset and `out_features` is unset, will default to the last stage. Must be in the
            same order as defined in the `stage_names` attribute.

    Example:
    ```python
    >>> from transformers import BitConfig, BitModel

    >>> # Initializing a BiT bit-50 style configuration
    >>> configuration = BitConfig()

    >>> # Initializing a model (with random weights) from the bit-50 style configuration
    >>> model = BitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = ...
    layer_types = ...
    supported_padding = ...
    def __init__(self, num_channels=..., embedding_size=..., hidden_sizes=..., depths=..., layer_type=..., hidden_act=..., global_padding=..., num_groups=..., drop_path_rate=..., embedding_dynamic_padding=..., output_stride=..., width_factor=..., out_features=..., out_indices=..., **kwargs) -> None:
        ...
    


