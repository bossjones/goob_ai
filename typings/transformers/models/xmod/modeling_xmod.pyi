"""
This type stub file was generated by pyright.
"""

import torch
from typing import List, Optional, Tuple, Union
from torch import nn
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithCrossAttentions, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward
from .configuration_xmod import XmodConfig

"""PyTorch X-MOD model."""
logger = ...
class XmodEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=..., past_key_values_length=...): # -> Any:
        ...
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds): # -> Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        ...
    


class XmodSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class XmodSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        ...
    


class XmodAttention(nn.Module):
    def __init__(self, config, position_embedding_type=...) -> None:
        ...
    
    def prune_heads(self, heads): # -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    


class XmodIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class XmodAdapter(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class XmodOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        ...
    
    def lang_adapter(self, lang_ids: torch.Tensor, hidden_states: torch.Tensor): # -> Tensor:
        ...
    


class XmodLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, lang_ids: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor]:
        ...
    
    def feed_forward_chunk(self, attention_output): # -> Any:
        ...
    


class XmodEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, lang_ids: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        ...
    


class XmodPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class XmodPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XmodConfig
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    def set_default_language(self, language: str): # -> None:
        """
        Set the default language code for the model. This is used when the language is not specified in the input.

        Args:
            language (`str`): The language code, such as `"en_XX"` or `"de_DE"`.
        """
        ...
    
    def freeze_embeddings_and_language_adapters(self): # -> None:
        """
        Freeze the embeddings and language adapters of the model. Usually, this is applied before the model is
        fine-tuned on a downstream task.
        """
        ...
    


XMOD_START_DOCSTRING = ...
XMOD_INPUTS_DOCSTRING = ...
@add_start_docstrings("The bare X-MOD Model transformer outputting raw hidden-states without any specific head on top.", XMOD_START_DOCSTRING)
class XmodModel(XmodPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self): # -> Embedding:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.Tensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., token_type_ids: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., head_mask: Optional[torch.Tensor] = ..., inputs_embeds: Optional[torch.Tensor] = ..., encoder_hidden_states: Optional[torch.Tensor] = ..., encoder_attention_mask: Optional[torch.Tensor] = ..., past_key_values: Optional[List[torch.FloatTensor]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors:
        of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        ...
    


@add_start_docstrings("X-MOD Model with a `language modeling` head on top for CLM fine-tuning.", XMOD_START_DOCSTRING)
class XmodForCausalLM(XmodPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., past_key_values: Tuple[Tuple[torch.FloatTensor]] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns: `transformers.modeling_outputs.CausalLMOutputWithCrossAttentions` or `tuple(torch.FloatTensor)`

        Example:

        ```python
        >>> from transformers import AutoTokenizer, XmodForCausalLM, AutoConfig
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        >>> config = AutoConfig.from_pretrained("facebook/xmod-base")
        >>> config.is_decoder = True
        >>> model = XmodForCausalLM.from_pretrained("facebook/xmod-base", config=config)
        >>> model.set_default_language("en_XX")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.logits
        ```"""
        ...
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=..., attention_mask=..., **model_kwargs): # -> dict[str, Any]:
        ...
    


@add_start_docstrings("""X-MOD Model with a `language modeling` head on top.""", XMOD_START_DOCSTRING)
class XmodForMaskedLM(XmodPreTrainedModel):
    _tied_weights_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self): # -> Linear:
        ...
    
    def set_output_embeddings(self, new_embeddings): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., encoder_hidden_states: Optional[torch.FloatTensor] = ..., encoder_attention_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        ...
    


class XmodLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, features, **kwargs): # -> Any:
        ...
    


@add_start_docstrings("""
    X-MOD Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, XMOD_START_DOCSTRING)
class XmodForSequenceClassification(XmodPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""
    X-MOD Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, XMOD_START_DOCSTRING)
class XmodForMultipleChoice(XmodPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        ...
    


@add_start_docstrings("""
    X-MOD Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, XMOD_START_DOCSTRING)
class XmodForTokenClassification(XmodPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., labels: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        ...
    


class XmodClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, features, **kwargs): # -> Any:
        ...
    


@add_start_docstrings("""
    X-MOD Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, XMOD_START_DOCSTRING)
class XmodForQuestionAnswering(XmodPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(XMOD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., lang_ids: Optional[torch.LongTensor] = ..., attention_mask: Optional[torch.FloatTensor] = ..., token_type_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ..., start_positions: Optional[torch.LongTensor] = ..., end_positions: Optional[torch.LongTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        ...
    


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=...):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    ...

