"""
This type stub file was generated by pyright.
"""

from functools import lru_cache
from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast

"""Tokenization classes for Whisper."""
logger = ...
VOCAB_FILES_NAMES = ...
class WhisperTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" Whisper tokenizer (backed by HuggingFace's *tokenizers* library).

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            Path to the vocabulary file.
        merges_file (`str`, *optional*):
            Path to the merges file.
        normalizer_file (`str`, *optional*):
            Path to the normalizer_file file.
        tokenizer_file (`str`, *optional*):
            Path to [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token. The `decoder_start_token_id` is used to set the first token as
            `"<|startoftranscript|>"` when generating.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (Whisper tokenizer detect beginning of words by the preceding space).
        language (`str`, *optional*):
            The language of the transcription text. The corresponding language id token is appended to the start of the
            sequence for multilingual speech recognition and speech translation tasks, e.g. for Spanish the token
            `"<|es|>"` is appended to the start of sequence. This should be used for multilingual fine-tuning only.
        task (`str`, *optional*):
            Task identifier to append at the start of sequence (if any). This should be used for mulitlingual
            fine-tuning, with `"transcribe"` for speech recognition and `"translate"` for speech translation.
        predict_timestamps (`bool`, *optional*, defaults to `False`):
            Whether to omit the `<|notimestamps|>` token at the start of the sequence.
    """
    vocab_files_names = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file=..., merges_file=..., normalizer_file=..., tokenizer_file=..., unk_token=..., bos_token=..., eos_token=..., add_prefix_space=..., language=..., task=..., predict_timestamps=..., **kwargs) -> None:
        ...
    
    @lru_cache
    def timestamp_ids(self, time_precision=...): # -> int | List[int]:
        """
        Compute the timestamp token ids for a given precision and save to least-recently used (LRU) cache.

        Args:
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
        """
        ...
    
    def decode(self, token_ids, skip_special_tokens: bool = ..., clean_up_tokenization_spaces: bool = ..., output_offsets: bool = ..., time_precision: float = ..., decode_with_timestamps: bool = ..., normalize: bool = ..., basic_normalize: bool = ..., remove_diacritics: bool = ..., **kwargs) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            output_offsets (`bool`, *optional*, defaults to `False`):
                Whether or not to output the offsets of the tokens. This should only be set if the model predicted
                timestamps.
            time_precision (`float`, `optional`, defaults to 0.02):
                The time ratio to convert from token to time.
            decode_with_timestamps (`bool`, *optional*, defaults to `False`):
                Whether or not to decode with timestamps included in the raw text.
            normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the English text normalizer to the decoded text. Only applicable when the
                target text is in English. Otherwise, the basic text normalizer should be applied.
            basic_normalize (`bool`, *optional*, defaults to `False`):
                Whether or not to apply the Basic text normalizer to the decoded text. Applicable to multilingual
                target text.
            remove_diacritics (`bool`, *optional*, defaults to `False`):
                Whether or not to remove diacritics when applying the Basic text normalizer. Removing diacritics may
                destroy information in the decoded text, hence it should be used with caution.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.
        Returns:
            `str`: The decoded sentence.
        """
        ...
    
    def normalize(self, text): # -> str:
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which preforms commons transformation on
        english text.
        """
        ...
    
    @staticmethod
    def basic_normalize(text, remove_diacritics=...): # -> str:
        """
        Normalize a given string using the `BasicTextNormalizer` class, which preforms commons transformation on
        multilingual text.
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def set_prefix_tokens(self, language: str = ..., task: str = ..., predict_timestamps: bool = ...): # -> None:
        """
        Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to
        update the prefix tokens as required when fine-tuning. Example:

        ```python
        >>> # instantiate the tokenizer and set the prefix token to Spanish
        >>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
        >>> # now switch the prefix token from Spanish to French
        >>> tokenizer.set_prefix_tokens(language="french")
        ```

        Args:
            language (`str`, *optional*, defaults to `None`):
                The language of the transcription text.
            task (`str`, *optional*, defaults to `None`):
                Task identifier to append at the start of sequence (if any).
            predict_timestamps (`bool`, *optional*, defaults to `None`):
                Whether to omit the `<|notimestamps|>` token at the start of the sequence.
        """
        ...
    
    @property
    def prefix_tokens(self) -> List[int]:
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
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
    
    @property
    def default_chat_template(self): # -> LiteralString:
        """
        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        """
        ...
    
    def get_decoder_prompt_ids(self, task=..., language=..., no_timestamps=...): # -> list[tuple[int, int]]:
        ...
    
    def get_prompt_ids(self, text: str, return_tensors=...): # -> Any | EncodingFast:
        """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
        ...
    


