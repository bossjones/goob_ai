"""
This type stub file was generated by pyright.
"""

import torch
from typing import Any, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_longt5 import LongT5Config

""" PyTorch LongT5 model."""
logger = ...
_CONFIG_FOR_DOC = ...
_CHECKPOINT_FOR_DOC = ...
class LongT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        ...
    
    def forward(self, hidden_states):
        ...
    


LongT5LayerNorm = ...
class LongT5DenseActDense(nn.Module):
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LongT5DenseGatedActDense(nn.Module):
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def forward(self, hidden_states): # -> Any:
        ...
    


class LongT5LayerFF(nn.Module):
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class LongT5Attention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def compute_bias(self, query_length, key_length, device=...): # -> Any:
        """Compute binned relative position bias"""
        ...
    
    def forward(self, hidden_states, mask=..., key_value_states=..., position_bias=..., past_key_value=..., layer_head_mask=..., query_length=..., use_cache=..., output_attentions=...): # -> tuple[Any, tuple[Tensor | Any, Tensor | Any] | None, Any | Tensor, Any | Tensor] | tuple[Any, tuple[Tensor | Any, Tensor | Any] | None, Any | Tensor]:
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        ...
    


class LongT5LocalAttention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = ...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def compute_bias(self, block_length: int): # -> Any:
        """Compute binned relative position bias"""
        ...
    
    def forward(self, hidden_states, mask=..., position_bias=..., layer_head_mask=..., output_attentions=...): # -> tuple[Any, None, Tensor | Any, Tensor | Any] | tuple[Any, None, Tensor | Any]:
        ...
    


class LongT5TransientGlobalAttention(nn.Module):
    def __init__(self, config: LongT5Config, has_relative_attention_bias: bool = ...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def compute_bias(self, block_length: int): # -> Any:
        """Compute binned relative position bias"""
        ...
    
    def compute_side_bias(self, mask: torch.Tensor, global_segment_ids: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states, mask=..., position_bias=..., layer_head_mask=..., output_attentions=...): # -> tuple[Any, None, Tensor | Any, Tensor | Any] | tuple[Any, None, Tensor | Any]:
        ...
    


class LongT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=...): # -> Any:
        ...
    


class LongT5LayerLocalSelfAttention(nn.Module):
    """Local self attention used in encoder"""
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., output_attentions=..., **kwargs: Any): # -> Any:
        ...
    


class LongT5LayerTransientGlobalSelfAttention(nn.Module):
    """Transient-Global self attention used in encoder"""
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., layer_head_mask=..., output_attentions=..., **kwargs: Any): # -> Any:
        ...
    


class LongT5LayerCrossAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, key_value_states, attention_mask=..., position_bias=..., layer_head_mask=..., past_key_value=..., use_cache=..., query_length=..., output_attentions=...): # -> Any:
        ...
    


class LongT5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=...) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., position_bias=..., encoder_hidden_states=..., encoder_attention_mask=..., encoder_decoder_position_bias=..., layer_head_mask=..., cross_attn_layer_head_mask=..., past_key_value=..., use_cache=..., output_attentions=..., return_dict=...): # -> Any:
        ...
    


class LongT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongT5Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...
    


class LongT5Stack(LongT5PreTrainedModel):
    def __init__(self, config, embed_tokens=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., inputs_embeds=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


LONGT5_START_DOCSTRING = ...
LONGT5_INPUTS_DOCSTRING = ...
LONGT5_ENCODER_INPUTS_DOCSTRING = ...
__HEAD_MASK_WARNING_MSG = ...
@add_start_docstrings("The bare LONGT5 Model transformer outputting raw hidden-states without any specific head on top.", LONGT5_START_DOCSTRING)
class LongT5Model(LongT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_encoder(self): # -> LongT5Stack:
        ...
    
    def get_decoder(self): # -> LongT5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., decoder_head_mask: Optional[torch.FloatTensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., inputs_embeds: Optional[torch.Tensor] = ..., decoder_inputs_embeds: Optional[torch.Tensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LongT5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5Model.from_pretrained("google/long-t5-local-base")

        >>> # Let's try a very long encoder input.
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1

        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...
    


@add_start_docstrings("""LONGT5 Model with a `language modeling` head on top.""", LONGT5_START_DOCSTRING)
class LongT5ForConditionalGeneration(LongT5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ...
    _tied_weights_keys = ...
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def get_encoder(self): # -> LongT5Stack:
        ...
    
    def get_decoder(self): # -> LongT5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(LONGT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.BoolTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., decoder_head_mask: Optional[torch.FloatTensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = ..., past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
        >>> model = LongT5ForConditionalGeneration.from_pretrained(
        ...     "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
        ... )

        >>> # Let's try a very long input.
        >>> inputs = tokenizer(100 * "studies have shown that owning a dog is good for you ", return_tensors="pt")
        >>> input_ids = inputs.input_ids

        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        abstractthe aim of this article is to provide an overview of the literature on the role of dog
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor): # -> Tensor:
        ...
    


@add_start_docstrings("The bare LONGT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.", LONGT5_START_DOCSTRING)
class LongT5EncoderModel(LongT5PreTrainedModel):
    _tied_weights_keys = ...
    _keys_to_ignore_on_load_unexpected = ...
    def __init__(self, config: LongT5Config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, new_embeddings): # -> None:
        ...
    
    def get_encoder(self): # -> LongT5Stack:
        ...
    
    @add_start_docstrings_to_model_forward(LONGT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")
        >>> model = LongT5EncoderModel.from_pretrained("google/long-t5-local-base")
        >>> input_ids = tokenizer(
        ...     100 * "Studies have been shown that owning a dog is good for you ", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        ...
    


