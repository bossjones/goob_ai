"""
This type stub file was generated by pyright.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Tuple
from flax.core.frozen_dict import FrozenDict
from ...modeling_flax_utils import FlaxPreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_albert import AlbertConfig

logger = ...
_CHECKPOINT_FOR_DOC = ...
_CONFIG_FOR_DOC = ...
@flax.struct.dataclass
class FlaxAlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`FlaxAlbertForPreTraining`].

    Args:
        prediction_logits (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`jnp.ndarray` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(jnp.ndarray)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape
            `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(jnp.ndarray)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    prediction_logits: jnp.ndarray = ...
    sop_logits: jnp.ndarray = ...
    hidden_states: Optional[Tuple[jnp.ndarray]] = ...
    attentions: Optional[Tuple[jnp.ndarray]] = ...


ALBERT_START_DOCSTRING = ...
ALBERT_INPUTS_DOCSTRING = ...
class FlaxAlbertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, token_type_ids, position_ids, deterministic: bool = ...):
        ...
    


class FlaxAlbertSelfAttention(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic=..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FlaxAlbertLayer(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ...): # -> tuple[Any, Any] | tuple[Any]:
        ...
    


class FlaxAlbertLayerCollection(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ...): # -> tuple[Any | tuple[()] | tuple[Any], ...] | tuple[Any, ...] | tuple[Any, tuple[()] | tuple[Any]] | tuple[Any]:
        ...
    


class FlaxAlbertLayerCollections(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    layer_index: Optional[str] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ...): # -> tuple[Any | tuple[()] | tuple[Any], ...] | tuple[Any, ...] | tuple[Any, tuple[()] | tuple[Any]] | tuple[Any]:
        ...
    


class FlaxAlbertLayerGroups(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxAlbertEncoder(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, attention_mask, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, ...] | FlaxBaseModelOutput:
        ...
    


class FlaxAlbertOnlyMLMHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    bias_init: Callable[..., np.ndarray] = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, hidden_states, shared_embedding=...):
        ...
    


class FlaxAlbertSOPHead(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, pooled_output, deterministic=...):
        ...
    


class FlaxAlbertPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlbertConfig
    base_model_prefix = ...
    module_class: nn.Module = ...
    def __init__(self, config: AlbertConfig, input_shape: Tuple = ..., seed: int = ..., dtype: jnp.dtype = ..., _do_init: bool = ..., **kwargs) -> None:
        ...
    
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = ...) -> FrozenDict:
        ...
    
    @add_start_docstrings_to_model_forward(ALBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(self, input_ids, attention_mask=..., token_type_ids=..., position_ids=..., params: dict = ..., dropout_rng: jax.random.PRNGKey = ..., train: bool = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...):
        ...
    


class FlaxAlbertModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    add_pooling_layer: bool = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids: Optional[np.ndarray] = ..., position_ids: Optional[np.ndarray] = ..., deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any, ...]] | Any | tuple[Any, Any, *tuple[Any, ...]] | FlaxBaseModelOutputWithPooling:
        ...
    


@add_start_docstrings("The bare Albert Model transformer outputting raw hidden-states without any specific head on top.", ALBERT_START_DOCSTRING)
class FlaxAlbertModel(FlaxAlbertPreTrainedModel):
    module_class = ...


class FlaxAlbertForPreTrainingModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, Any, *tuple[Any, ...]] | Any | FlaxAlbertForPreTrainingOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `sentence order prediction (classification)` head.
    """, ALBERT_START_DOCSTRING)
class FlaxAlbertForPreTraining(FlaxAlbertPreTrainedModel):
    module_class = ...


FLAX_ALBERT_FOR_PRETRAINING_DOCSTRING = ...
class FlaxAlbertForMaskedLMModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any, ...]] | Any | tuple[Any, Any, *tuple[Any, ...]] | FlaxMaskedLMOutput:
        ...
    


@add_start_docstrings("""Albert Model with a `language modeling` head on top.""", ALBERT_START_DOCSTRING)
class FlaxAlbertForMaskedLM(FlaxAlbertPreTrainedModel):
    module_class = ...


class FlaxAlbertForSequenceClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any, ...]] | Any | FlaxSequenceClassifierOutput:
        ...
    


@add_start_docstrings("""
    Albert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, ALBERT_START_DOCSTRING)
class FlaxAlbertForSequenceClassification(FlaxAlbertPreTrainedModel):
    module_class = ...


class FlaxAlbertForMultipleChoiceModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any, ...]] | Any | FlaxMultipleChoiceModelOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, ALBERT_START_DOCSTRING)
class FlaxAlbertForMultipleChoice(FlaxAlbertPreTrainedModel):
    module_class = ...


class FlaxAlbertForTokenClassificationModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, *tuple[Any, ...]] | Any | tuple[Any, Any, *tuple[Any, ...]] | FlaxTokenClassifierOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, ALBERT_START_DOCSTRING)
class FlaxAlbertForTokenClassification(FlaxAlbertPreTrainedModel):
    module_class = ...


class FlaxAlbertForQuestionAnsweringModule(nn.Module):
    config: AlbertConfig
    dtype: jnp.dtype = ...
    def setup(self): # -> None:
        ...
    
    def __call__(self, input_ids, attention_mask, token_type_ids, position_ids, deterministic: bool = ..., output_attentions: bool = ..., output_hidden_states: bool = ..., return_dict: bool = ...): # -> tuple[Any, Any, *tuple[Any, ...]] | Any | tuple[Any, Any, Any, *tuple[Any, ...]] | FlaxQuestionAnsweringModelOutput:
        ...
    


@add_start_docstrings("""
    Albert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, ALBERT_START_DOCSTRING)
class FlaxAlbertForQuestionAnswering(FlaxAlbertPreTrainedModel):
    module_class = ...


