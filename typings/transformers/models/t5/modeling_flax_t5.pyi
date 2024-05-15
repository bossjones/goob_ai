"""
This type stub file was generated by pyright.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPastAndCrossAttentions, FlaxCausalLMOutputWithCrossAttentions
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_t5 import T5Config

""" Flax T5 model."""
logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
remat = ...
def shift_tokens_right(input_ids: jnp.ndarray, pad_token_id: int, decoder_start_token_id: int) -> jnp.ndarray:
    """
    Shift input ids one token to the right.
    """
    ...

class FlaxT5LayerNorm(nn.Module):
    hidden_size: int
    dtype: jnp.dtype = ...
    eps: float = ...
    weight_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        """
        Construct a layernorm module in the T5 style; No bias and no subtraction of mean.
        """
        ...
    


class FlaxT5DenseActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=...):
        ...
    


class FlaxT5DenseGatedActDense(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic):
        ...
    


class FlaxT5LayerFF(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, deterministic=...):
        ...
    


class FlaxT5Attention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    causal: bool = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        ...
    
    def __call__(self, hidden_states, attention_mask=..., key_value_states=..., position_bias=..., use_cache=..., output_attentions=..., deterministic=..., init_cache=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ...
    


class FlaxT5LayerSelfAttention(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., position_bias=..., output_attentions=..., deterministic=..., init_cache=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...
    


class FlaxT5LayerCrossAttention(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, key_value_states, attention_mask=..., position_bias=..., output_attentions=..., deterministic=...): # -> tuple[Any, Any, Any] | tuple[Any, Any]:
        ...
    


class FlaxT5Block(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool = ...
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., output_attentions=..., return_dict=..., deterministic=..., init_cache=...): # -> tuple[tuple[Any, Any, Any] | tuple[Any, Any], Any, Any] | tuple[tuple[Any, Any, Any] | tuple[Any, Any], Any]:
        ...
    


class FlaxT5LayerCollection(nn.Module):
    config: T5Config
    has_relative_attention_bias: bool
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., output_attentions=..., deterministic=..., init_cache=...): # -> tuple[tuple[Any, Any, Any] | tuple[Any, Any], Any, Any] | tuple[tuple[Any, Any, Any] | tuple[Any, Any], Any]:
        ...
    


class FlaxT5BlockCollection(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions: bool = ..., output_hidden_states: bool = ..., deterministic: bool = ..., init_cache: bool = ...): # -> FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxT5Stack(nn.Module):
    config: T5Config
    embed_tokens: nn.Embed
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ..., deterministic: bool = ..., init_cache: bool = ...): # -> Any | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


T5_ENCODE_INPUTS_DOCSTRING = ...
T5_DECODE_INPUTS_DOCSTRING = ...
T5_INPUTS_DOCSTRING = ...
class FlaxT5PreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: T5Config, input_shape: Tuple[int] = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., gradient_checkpointing: bool = ..., **kwargs) -> None:
        ...
    
    def enable_gradient_checkpointing(self): # -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., decoder_input_ids: jnp.ndarray = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...
    
    def init_cache(self, batch_size, max_length, encoder_outputs):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
            encoder_outputs (`Union[FlaxBaseModelOutput, tuple(tuple(jnp.ndarray)]`):
                `encoder_outputs` consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*:
                `attentions`). `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
                is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
                cross-attention of the decoder.
        """
        ...
    
    @add_start_docstrings(T5_ENCODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutput, config_class=T5Config)
    def encode(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)
        ```"""
        ...
    
    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxBaseModelOutputWithPastAndCrossAttentions, config_class=T5Config)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
        >>> import jax.numpy as jnp

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> text = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        ...
    


T5_START_DOCSTRING = ...
@add_start_docstrings("The bare T5 Model transformer outputting raw hidden-stateswithout any specific head on top.", T5_START_DOCSTRING)
class FlaxT5Module(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., encoder_outputs=..., output_attentions=..., output_hidden_states=..., return_dict=..., deterministic: bool = ...): # -> FlaxSeq2SeqModelOutput:
        ...
    


class FlaxT5Model(FlaxT5PreTrainedModel):
    module_class = ...


FLAX_T5_MODEL_DOCSTRING = ...
@add_start_docstrings("The bare T5 Model transformer outputting encoder's raw hidden-states without any specific head on top.", T5_START_DOCSTRING)
class FlaxT5EncoderModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids=..., attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict: bool = ..., deterministic: bool = ...): # -> Any | FlaxBaseModelOutputWithPastAndCrossAttentions:
        ...
    


class FlaxT5EncoderModel(FlaxT5PreTrainedModel):
    module_class = ...
    @add_start_docstrings_to_model_forward(T5_ENCODE_INPUTS_DOCSTRING)
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...):
        ...
    


@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class FlaxT5ForConditionalGenerationModule(nn.Module):
    config: T5Config
    dtype: jnp.dtype = ...
    gradient_checkpointing: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids=..., attention_mask=..., decoder_input_ids=..., decoder_attention_mask=..., encoder_outputs=..., output_attentions=..., output_hidden_states=..., return_dict=..., deterministic: bool = ...): # -> Any | FlaxSeq2SeqLMOutput:
        ...
    


class FlaxT5ForConditionalGeneration(FlaxT5PreTrainedModel):
    module_class = ...
    @add_start_docstrings(T5_DECODE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FlaxCausalLMOutputWithCrossAttentions, config_class=T5Config)
    def decode(self, decoder_input_ids, encoder_outputs, encoder_attention_mask: Optional[jnp.ndarray] = ..., decoder_attention_mask: Optional[jnp.ndarray] = ..., past_key_values: dict = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ..., train: bool = ..., params: dict = ..., dropout_rng: PRNGKey = ...): # -> FlaxCausalLMOutputWithCrossAttentions | Any:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
        >>> import jax.numpy as jnp

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = FlaxT5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> text = "summarize: My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer(text, return_tensors="np")
        >>> encoder_outputs = model.encode(**inputs)

        >>> decoder_start_token_id = model.config.decoder_start_token_id
        >>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

        >>> outputs = model.decode(decoder_input_ids, encoder_outputs)
        >>> logits = outputs.logits
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, max_length, attention_mask: Optional[jax.Array] = ..., decoder_attention_mask: Optional[jax.Array] = ..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


FLAX_T5_CONDITIONAL_GENERATION_DOCSTRING = ...
