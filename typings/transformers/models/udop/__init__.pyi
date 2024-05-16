"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_sentencepiece_available, is_tokenizers_available, is_torch_available
from .configuration_udop import UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP, UdopConfig
from .processing_udop import UdopProcessor

_import_structure = ...
if not is_sentencepiece_available():
    ...
if not is_tokenizers_available():
    ...
if not is_torch_available():
    ...
if TYPE_CHECKING:
    ...
else:
    ...
