"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_end_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_blenderbot_small import BlenderbotSmallConfig

""" PyTorch BlenderbotSmall model."""
logger = ...
_CONFIG_FOR_DOC = ...
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int): # -> Tensor:
    """
    Shift input ids one token to the right.
    """
    ...

class BlenderbotSmallLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        ...
    
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = ...): # -> Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        ...
    


class BlenderbotSmallAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = ..., is_decoder: bool = ..., bias: bool = ..., is_causal: bool = ..., config: Optional[BlenderbotSmallConfig] = ...) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, key_value_states: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., output_attentions: bool = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class BlenderbotSmallEncoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.FloatTensor, layer_head_mask: torch.FloatTensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


BLENDERBOT_SMALL_ATTENTION_CLASSES = ...
class BlenderbotSmallDecoderLayer(nn.Module):
    def __init__(self, config: BlenderbotSmallConfig) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., layer_head_mask: Optional[torch.Tensor] = ..., cross_attn_layer_head_mask: Optional[torch.Tensor] = ..., past_key_value: Optional[Tuple[torch.Tensor]] = ..., output_attentions: Optional[bool] = ..., use_cache: Optional[bool] = ...) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class BlenderbotSmallPreTrainedModel(PreTrainedModel):
    config_class = BlenderbotSmallConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    @property
    def dummy_inputs(self): # -> dict[str, Tensor]:
        ...
    


BLENDERBOT_SMALL_START_DOCSTRING = ...
BLENDERBOT_SMALL_GENERATION_EXAMPLE = ...
BLENDERBOT_SMALL_INPUTS_DOCSTRING = ...
class BlenderbotSmallEncoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotSmallEncoderLayer`].

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = ...) -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...): # -> tuple[Any, ...] | BaseModelOutput:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    


class BlenderbotSmallDecoder(BlenderbotSmallPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotSmallDecoderLayer`]

    Args:
        config: BlenderbotSmallConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BlenderbotSmallConfig, embed_tokens: Optional[nn.Embedding] = ...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def forward(self, input_ids=..., attention_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., head_mask=..., cross_attn_head_mask=..., past_key_values=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    


@add_start_docstrings("The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top.", BLENDERBOT_SMALL_START_DOCSTRING)
class BlenderbotSmallModel(BlenderbotSmallPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config: BlenderbotSmallConfig) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_encoder(self): # -> BlenderbotSmallEncoder:
        ...
    
    def get_decoder(self): # -> BlenderbotSmallDecoder:
        ...
    
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., decoder_head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.Tensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BlenderbotSmallModel

        >>> model = BlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

        >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
        >>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")  # Batch size 1
        >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 3, 512]
        ```"""
        ...
    


@add_start_docstrings("The BlenderbotSmall Model with a language modeling head. Can be used for summarization.", BLENDERBOT_SMALL_START_DOCSTRING)
class BlenderbotSmallForConditionalGeneration(BlenderbotSmallPreTrainedModel):
    base_model_prefix = ...
    _keys_to_ignore_on_load_missing = ...
    _tied_weights_keys = ...
    def __init__(self, config: BlenderbotSmallConfig) -> None:
        ...
    
    def get_encoder(self): # -> BlenderbotSmallEncoder:
        ...
    
    def get_decoder(self): # -> BlenderbotSmallDecoder:
        ...
    
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = ...) -> nn.Embedding:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(BLENDERBOT_SMALL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BLENDERBOT_SMALL_GENERATION_EXAMPLE)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., decoder_input_ids: Optional[torch.LongTensor] = ..., decoder_attention_mask: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.Tensor] = ..., decoder_head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.Tensor] = ..., decoder_inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        ...
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=..., attention_mask=..., head_mask=..., decoder_head_mask=..., cross_attn_head_mask=..., use_cache=..., encoder_outputs=..., **kwargs): # -> dict[str, Any]:
        ...
    


class BlenderbotSmallDecoderWrapper(BlenderbotSmallPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, *args, **kwargs): # -> Any:
        ...
    


class BlenderbotSmallForCausalLM(BlenderbotSmallPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding | Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    def set_decoder(self, decoder): # -> None:
        ...
    
    def get_decoder(self): # -> BlenderbotSmallDecoder:
        ...
    
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: torch.LongTensor = ..., attention_mask: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.Tensor] = ..., cross_attn_head_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BlenderbotSmallForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
        >>> model = BlenderbotSmallForCausalLM.from_pretrained("facebook/blenderbot_small-90M", add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., use_cache=..., **kwargs): # -> dict[str, Any]:
        ...
    


