"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

""" CvT model configuration"""
logger = ...
class CvtConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CvtModel`]. It is used to instantiate a CvT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CvT
    [microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3]`):
            The kernel size of each encoder's patch embedding.
        patch_stride (`List[int]`, *optional*, defaults to `[4, 2, 2]`):
            The stride size of each encoder's patch embedding.
        patch_padding (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            The padding size of each encoder's patch embedding.
        embed_dim (`List[int]`, *optional*, defaults to `[64, 192, 384]`):
            Dimension of each of the encoder blocks.
        num_heads (`List[int]`, *optional*, defaults to `[1, 3, 6]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        depth (`List[int]`, *optional*, defaults to `[1, 2, 10]`):
            The number of layers in each encoder block.
        mlp_ratios (`List[float]`, *optional*, defaults to `[4.0, 4.0, 4.0, 4.0]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_drop_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
            The dropout ratio for the attention probabilities.
        drop_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.0]`):
            The dropout ratio for the patch embeddings probabilities.
        drop_path_rate (`List[float]`, *optional*, defaults to `[0.0, 0.0, 0.1]`):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        qkv_bias (`List[bool]`, *optional*, defaults to `[True, True, True]`):
            The bias bool for query, key and value in attentions
        cls_token (`List[bool]`, *optional*, defaults to `[False, False, True]`):
            Whether or not to add a classification token to the output of each of the last 3 stages.
        qkv_projection_method (`List[string]`, *optional*, defaults to ["dw_bn", "dw_bn", "dw_bn"]`):
            The projection method for query, key and value Default is depth-wise convolutions with batch norm. For
            Linear projection use "avg".
        kernel_qkv (`List[int]`, *optional*, defaults to `[3, 3, 3]`):
            The kernel size for query, key and value in attention layer
        padding_kv (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The padding size for key and value in attention layer
        stride_kv (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            The stride size for key and value in attention layer
        padding_q (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The padding size for query in attention layer
        stride_q (`List[int]`, *optional*, defaults to `[1, 1, 1]`):
            The stride size for query in attention layer
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.

    Example:

    ```python
    >>> from transformers import CvtConfig, CvtModel

    >>> # Initializing a Cvt msft/cvt style configuration
    >>> configuration = CvtConfig()

    >>> # Initializing a model (with random weights) from the msft/cvt style configuration
    >>> model = CvtModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, num_channels=..., patch_sizes=..., patch_stride=..., patch_padding=..., embed_dim=..., num_heads=..., depth=..., mlp_ratio=..., attention_drop_rate=..., drop_rate=..., drop_path_rate=..., qkv_bias=..., cls_token=..., qkv_projection_method=..., kernel_qkv=..., padding_kv=..., stride_kv=..., padding_q=..., stride_q=..., initializer_range=..., layer_norm_eps=..., **kwargs) -> None:
        ...
    


