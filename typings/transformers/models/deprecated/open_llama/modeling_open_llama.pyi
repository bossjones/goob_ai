"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_open_llama import OpenLlamaConfig

""" PyTorch Open-Llama model."""
logger = ...
_CONFIG_FOR_DOC = ...
class OpenLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=...) -> None:
        """
        OpenLlamaRMSNorm is equivalent to T5LayerNorm
        """
        ...
    
    def forward(self, hidden_states):
        ...
    


class OpenLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=..., base=..., device=...) -> None:
        ...
    
    def forward(self, x, seq_len=...): # -> tuple[Any, Any]:
        ...
    


class OpenLlamaLinearScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...) -> None:
        ...
    


class OpenLlamaDynamicNTKScalingRotaryEmbedding(OpenLlamaRotaryEmbedding):
    """OpenLlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=..., base=..., device=..., scaling_factor=...) -> None:
        ...
    


def rotate_half(x): # -> Tensor:
    """Rotates half the hidden dims of the input."""
    ...

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=...): # -> tuple[Any, Any]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    ...

class OpenLlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str, dropout_prob: float) -> None:
        ...
    
    def forward(self, x): # -> Any:
        ...
    


class OpenLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: OpenLlamaConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: bool = ..., use_cache: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ...
    


class OpenLlamaDecoderLayer(nn.Module):
    def __init__(self, config: OpenLlamaConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ...) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        ...
    


OPEN_LLAMA_START_DOCSTRING = ...
@add_start_docstrings("The bare Open-Llama Model outputting raw hidden-states without any specific head on top.", OPEN_LLAMA_START_DOCSTRING)
class OpenLlamaPreTrainedModel(PreTrainedModel):
    config_class = OpenLlamaConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...


OPEN_LLAMA_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare Open-Llama Model outputting raw hidden-states without any specific head on top.", OPEN_LLAMA_START_DOCSTRING)
class OpenLlamaModel(OpenLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OpenLlamaDecoderLayer`]

    Args:
        config: OpenLlamaConfig
    """
    def __init__(self, config: OpenLlamaConfig) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPast]:
        ...
    


class OpenLlamaForCausalLM(OpenLlamaPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear | None:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_decoder(self, decoder): # -> None:
        ...
    
    def get_decoder(self): # -> OpenLlamaModel:
        ...
    
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OpenLlamaForCausalLM

        >>> model = OpenLlamaForCausalLM.from_pretrained("openlm-research/open_llama_7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_7b")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., inputs_embeds=..., **kwargs): # -> dict[str, Any]:
        ...
    


@add_start_docstrings("""
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`OpenLlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """, OPEN_LLAMA_START_DOCSTRING)
class OpenLlamaForSequenceClassification(OpenLlamaPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OPEN_LLAMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.LongTensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


