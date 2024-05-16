"""
This type stub file was generated by pyright.
"""

import os
from typing import Union
from ...configuration_utils import PretrainedConfig

""" CLAP model configuration"""
logger = ...
class ClapTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapTextModel`]. It is used to instantiate a CLAP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLAP
    [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ClapTextModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`,
            `"relu"`, `"silu"` and `"relu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`ClapTextModel`].
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        projection_dim (`int`, *optional*, defaults to 512)
            Dimension of the projection head of the `ClapTextModelWithProjection`.

    Examples:

    ```python
    >>> from transformers import ClapTextConfig, ClapTextModel

    >>> # Initializing a CLAP text configuration
    >>> configuration = ClapTextConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ClapTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_factor=..., layer_norm_eps=..., projection_dim=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., position_embedding_type=..., use_cache=..., projection_hidden_act=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        ...
    


class ClapAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ClapAudioModel`]. It is used to instantiate a
    CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        window_size (`int`, *optional*, defaults to 8):
            Image size of the spectrogram
        num_mel_bins (`int`, *optional*, defaults to 64):
            Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
        spec_size (`int`, *optional*, defaults to 256):
            Desired input size of the spectrogram that the model supports. It can be different from the output of the
            `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size`
            of the audio models.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        patch_size (`int`, *optional*, defaults to 4):
            Patch size for the audio spectrogram
        patch_stride (`list`, *optional*, defaults to `[4, 4]`):
            Patch stride for the audio spectrogram
        num_classes (`int`, *optional*, defaults to 527):
            Number of classes used for the head training
        hidden_size (`int`, *optional*, defaults to 768):
            Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer's
            output,which is sent to the projection MLP layer.
        projection_dim (`int`, *optional*, defaults to 512):
            Hidden size of the projection layer.
        depths (`list`, *optional*, defaults to `[2, 2, 6, 2]`):
            Depths used for the Swin Layers of the audio model
        num_attention_heads (`list`, *optional*, defaults to `[4, 8, 16, 32]`):
            Number of attention heads used for the Swin Layers of the audio model
        enable_fusion (`bool`, *optional*, defaults to `False`):
            Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the
            best results.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the encoder.
        fusion_type (`[type]`, *optional*):
            Fusion type used for the patch fusion.
        patch_embed_input_channels (`int`, *optional*, defaults to 1):
            Number of channels used for the input spectrogram
        flatten_patch_embeds (`bool`, *optional*, defaults to `True`):
            Whether or not to flatten the patch embeddings
        patch_embeds_hidden_size (`int`, *optional*, defaults to 96):
            Hidden size of the patch embeddings. It is used as the number of output channels.
        enable_patch_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether or not to enable layer normalization for the patch embeddings
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Drop path rate for the patch fusion
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to add a bias to the query, key, value projections.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the mlp hidden dim to embedding dim.
        aff_block_r (`int`, *optional*, defaults to 4):
            downsize_ratio used in the AudioFF block
        num_hidden_layers (`int`, *optional*, defaults to 4):
            Number of hidden layers in the Transformer encoder.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        layer_norm_eps (`[type]`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from transformers import ClapAudioConfig, ClapAudioModel

    >>> # Initializing a ClapAudioConfig with laion/clap-htsat-fused style configuration
    >>> configuration = ClapAudioConfig()

    >>> # Initializing a ClapAudioModel (with random weights) from the laion/clap-htsat-fused style configuration
    >>> model = ClapAudioModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, window_size=..., num_mel_bins=..., spec_size=..., hidden_act=..., patch_size=..., patch_stride=..., num_classes=..., hidden_size=..., projection_dim=..., depths=..., num_attention_heads=..., enable_fusion=..., hidden_dropout_prob=..., fusion_type=..., patch_embed_input_channels=..., flatten_patch_embeds=..., patch_embeds_hidden_size=..., enable_patch_layer_norm=..., drop_path_rate=..., attention_probs_dropout_prob=..., qkv_bias=..., mlp_ratio=..., aff_block_r=..., num_hidden_layers=..., projection_hidden_act=..., layer_norm_eps=..., initializer_factor=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        ...
    


class ClapConfig(PretrainedConfig):
    r"""
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and audio projection layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""
    model_type = ...
    def __init__(self, text_config=..., audio_config=..., logit_scale_init_value=..., projection_dim=..., projection_hidden_act=..., initializer_factor=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs): # -> Self:
        r"""
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """
        ...
    


