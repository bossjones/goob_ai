"""
This type stub file was generated by pyright.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_mistral import MistralConfig

""" Flax Mistral model."""
logger = ...
_CONFIG_FOR_DOC = ...
_REAL_CHECKPOINT_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
MISTRAL_START_DOCSTRING = ...
MISTRAL_INPUTS_DOCSTRING = ...
class FlaxMistralRMSNorm(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


class FlaxMistralRotaryEmbedding(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, key, query, position_ids): # -> tuple[Any, Any]:
        ...
    


class FlaxMistralMLP(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states):
        ...
    


def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    ...

def create_sinusoidal_positions(num_pos, dim):
    ...

def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    ...

class FlaxMistralAttention(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = ..., position_ids: Optional[jnp.ndarray] = ..., deterministic: bool = ..., output_attentions: bool = ..., init_cache: bool = ...) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...
    


class FlaxMistralDecoderLayer(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ...): # -> tuple[Any, Any]:
        ...
    


class FlaxMistralPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MistralConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: MistralConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        ...
    
    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def __call__(self, input_ids, attention_mask=..., position_ids=..., params: dict = ..., past_key_values: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxMistralLayerCollection(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, tuple[()] | Any | None, tuple[()] | Any | None]:
        ...
    


class FlaxMistralModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., position_ids=..., deterministic=..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any | tuple[()], ...] | FlaxBaseModelOutput:
        ...
    


@add_start_docstrings("The bare Mistral Model transformer outputting raw hidden-states without any specific head on top.", MISTRAL_START_DOCSTRING)
class FlaxMistralModel(FlaxMistralPreTrainedModel):
    module_class = ...


class FlaxMistralForCausalLMModule(nn.Module):
    config: MistralConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask=..., position_ids=..., deterministic: bool = ..., init_cache: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any | tuple[()], ...]] | Any | FlaxCausalLMOutput:
        ...
    


@add_start_docstrings("""
    The Mistral Model transformer with a language modeling head (linear layer) on top.
    """, MISTRAL_START_DOCSTRING)
class FlaxMistralForCausalLM(FlaxMistralPreTrainedModel):
    module_class = ...
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = ...): # -> dict[str, Any]:
        ...
    
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        ...
    


