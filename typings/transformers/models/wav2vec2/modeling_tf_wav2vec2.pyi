"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFPreTrainedModel, keras, keras_serializable, unpack_inputs
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_wav2vec2 import Wav2Vec2Config

""" TensorFlow Wav2Vec2 model."""
logger = ...
_HIDDEN_STATES_START_POSITION = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
LARGE_NEGATIVE = ...
@dataclass
class TFWav2Vec2BaseModelOutput(ModelOutput):
    """
    Output type of [`TFWav2Vec2BaseModelOutput`], with potential hidden states and attentions.

    Args:
        last_hidden_state (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`tf.Tensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(tf.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(tf.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: tf.Tensor = ...
    extract_features: tf.Tensor = ...
    hidden_states: Tuple[tf.Tensor] | None = ...
    attentions: Tuple[tf.Tensor] | None = ...


class TFWav2Vec2GroupNorm(keras.layers.Layer):
    """
    From tensorflow-addons https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization
    """
    def __init__(self, groups: int = ..., axis: int = ..., epsilon: float = ..., center: bool = ..., scale: bool = ..., beta_initializer: keras.initializers.Initializer = ..., gamma_initializer: keras.initializers.Initializer = ..., beta_regularizer: keras.regularizers.Regularizer = ..., gamma_regularizer: keras.regularizers.Regularizer = ..., beta_constraint: keras.constraints.Constraint = ..., gamma_constraint: keras.constraints.Constraint = ..., **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    
    def get_config(self): # -> dict[str, Any]:
        ...
    
    def compute_output_shape(self, input_shape):
        ...
    


class TFWav2Vec2WeightNormConv1D(keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""
    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs) -> None:
        ...
    
    def build(self, input_shape): # -> None:
        ...
    
    def call(self, inputs):
        ...
    


class TFWav2Vec2NoLayerNormConvLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2LayerNormConvLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2GroupNormConvLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = ..., **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2PositionalConvEmbedding(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2SamePadLayer(keras.layers.Layer):
    def __init__(self, num_conv_pos_embeddings, **kwargs) -> None:
        ...
    
    def call(self, hidden_states):
        ...
    


class TFWav2Vec2FeatureEncoder(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        ...
    
    def call(self, input_values):
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2FeatureExtractor(TFWav2Vec2FeatureEncoder):
    def __init__(self, config, **kwargs) -> None:
        ...
    


class TFWav2Vec2FeatureProjection(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2Attention(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, key_value_states: tf.Tensor | None = ..., past_key_value: Tuple[Tuple[tf.Tensor]] | None = ..., attention_mask: tf.Tensor | None = ..., layer_head_mask: tf.Tensor | None = ..., training: Optional[bool] = ...) -> Tuple[tf.Tensor, tf.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2FeedForward(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, training: bool = ...) -> tf.Tensor:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2EncoderLayer(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2EncoderLayerStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., training: bool = ...) -> Tuple[tf.Tensor]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2Encoder(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2EncoderStableLayerNorm(keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@keras_serializable
class TFWav2Vec2MainLayer(keras.layers.Layer):
    config_class = Wav2Vec2Config
    def __init__(self, config: Wav2Vec2Config, **kwargs) -> None:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None = ..., token_type_ids: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., head_mask: tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ..., **kwargs: Any): # -> TFWav2Vec2BaseModelOutput:
        ...
    


class TFWav2Vec2PreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Wav2Vec2Config
    base_model_prefix = ...
    main_input_name = ...
    @property
    def input_signature(self): # -> dict[str, Any]:
        ...
    
    @property
    def dummy_inputs(self): # -> dict[str, Any]:
        ...
    
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    


WAV_2_VEC_2_START_DOCSTRING = ...
WAV_2_VEC_2_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.", WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2Model(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None = ..., token_type_ids: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., head_mask: tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: bool = ...) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, TFWav2Vec2Model
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


@add_start_docstrings("""TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""", WAV_2_VEC_2_START_DOCSTRING)
class TFWav2Vec2ForCTC(TFWav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config, *inputs, **kwargs) -> None:
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        ...
    
    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...
    
    @unpack_inputs
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None = ..., token_type_ids: tf.Tensor | None = ..., position_ids: tf.Tensor | None = ..., head_mask: tf.Tensor | None = ..., inputs_embeds: tf.Tensor | None = ..., output_attentions: Optional[bool] = ..., labels: tf.Tensor | None = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., training: Optional[bool] = ...) -> Union[TFCausalLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_values` docstring) Tokens with indices set to `-100` are ignored (masked),
            the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:

        Example:

        ```python
        >>> import tensorflow as tf
        >>> from transformers import AutoProcessor, TFWav2Vec2ForCTC
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        >>> model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  # Batch size 1
        >>> logits = model(input_values).logits
        >>> predicted_ids = tf.argmax(logits, axis=-1)

        >>> transcription = processor.decode(predicted_ids[0])

        >>> # compute loss
        >>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

        >>> # Pass transcription as `text` to encode labels
        >>> labels = processor(text=transcription, return_tensors="tf").input_ids

        >>> loss = model(input_values, labels=labels).loss
        ```"""
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


class TFWav2Vec2ForSequenceClassification(TFWav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def freeze_feature_extractor(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        ...
    
    def freeze_feature_encoder(self): # -> None:
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        ...
    
    def freeze_base_model(self): # -> None:
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        ...
    
    @unpack_inputs
    def call(self, input_values: tf.Tensor, attention_mask: tf.Tensor | None = ..., output_attentions: bool | None = ..., output_hidden_states: bool | None = ..., return_dict: bool | None = ..., labels: tf.Tensor | None = ..., training: bool = ...) -> TFSequenceClassifierOutput | Tuple[tf.Tensor]:
        ...
    
    def build(self, input_shape=...): # -> None:
        ...
    


