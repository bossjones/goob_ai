"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import TextInput

""" Tokenization class for SigLIP model."""
if TYPE_CHECKING:
    ...
logger = ...
VOCAB_FILES_NAMES = ...
SPIECE_UNDERLINE = ...
class SiglipTokenizer(PreTrainedTokenizer):
    """
    Construct a Siglip tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a *.spm* extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        eos_token (`str`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (`str`, *optional*, defaults to `"</s>"`):
            The token used for padding, for example when batching sequences of different lengths.
        additional_special_tokens (`List[str]`, *optional*):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        model_max_length (`int`, *optional*, defaults to 64):
            The maximum length (in number of tokens) for model inputs.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
    """
    vocab_files_names = ...
    model_input_names = ...
    def __init__(self, vocab_file, eos_token=..., unk_token=..., pad_token=..., additional_special_tokens=..., sp_model_kwargs: Optional[Dict[str, Any]] = ..., model_max_length=..., do_lower_case=..., **kwargs) -> None:
        ...
    
    def get_spm_processor(self): # -> SentencePieceProcessor:
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self): # -> dict[str, int]:
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. T5 does not make
        use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of zeros.
        """
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, d): # -> None:
        ...
    
    def remove_punctuation(self, text: str) -> str:
        ...
    
    def canonicalize_text(self, text, *, keep_punctuation_exact_string=...): # -> str:
        """Returns canonicalized `text` (puncuation removed).

        Args:
            text (`str`):
                String to be canonicalized.
            keep_punctuation_exact_string (`str`, *optional*):
                If provided, then this exact string is kept. For example providing '{}' will keep any occurrences of '{}'
                (but will still remove '{' and '}' that appear separately).
        """
        ...
    
    def tokenize(self, text: TextInput, add_special_tokens=..., **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens.
        """
        ...
    
    @property
    def unk_token_length(self): # -> int:
        ...
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    


