"""
This type stub file was generated by pyright.
"""

from torchvision.extension import _check_cuda_version

__version__ = ...
git_version = ...
if _check_cuda_version() > 0:
    cuda = ...
