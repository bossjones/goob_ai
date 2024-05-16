"""
This type stub file was generated by pyright.
"""

from ...configuration_utils import PretrainedConfig

""" VisualBERT model configuration"""
logger = ...
class VisualBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VisualBertModel`]. It is used to instantiate an
    VisualBERT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the VisualBERT
    [uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`VisualBertModel`]. Vocabulary size of the model. Defines the
            different tokens that can be represented by the `inputs_ids` passed to the forward method of
            [`VisualBertModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        visual_embedding_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the visual embeddings to be passed to the model.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`VisualBertModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bypass_transformer (`bool`, *optional*, defaults to `False`):
            Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
            model directly concatenates the visual embeddings from [`VisualBertEmbeddings`] with text output from
            transformers, and then pass it to a self-attention layer.
        special_visual_initialize (`bool`, *optional*, defaults to `True`):
            Whether or not the visual token type and position type embedding weights should be initialized the same as
            the textual token type and positive type embeddings. When set to `True`, the weights of the textual token
            type and position type embeddings are copied to the respective visual embedding layers.


    Example:

    ```python
    >>> from transformers import VisualBertConfig, VisualBertModel

    >>> # Initializing a VisualBERT visualbert-vqa-coco-pre style configuration
    >>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

    >>> # Initializing a model (with random weights) from the visualbert-vqa-coco-pre style configuration
    >>> model = VisualBertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = ...
    def __init__(self, vocab_size=..., hidden_size=..., visual_embedding_dim=..., num_hidden_layers=..., num_attention_heads=..., intermediate_size=..., hidden_act=..., hidden_dropout_prob=..., attention_probs_dropout_prob=..., max_position_embeddings=..., type_vocab_size=..., initializer_range=..., layer_norm_eps=..., bypass_transformer=..., special_visual_initialize=..., pad_token_id=..., bos_token_id=..., eos_token_id=..., **kwargs) -> None:
        ...
    


