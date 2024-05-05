# """ goob_ai.backend.ml.devices """
# from __future__ import annotations

# # SOURCE: https://github.com/socialhourmobile/SD-hassan-ns/blob/3b6b266b17e0fd0a9b17374cd2afbf4c59b7c245/modules/devices.py
# from typing import Optional, Union

# import torch

# from goob_ai.backend.ml import errors


# # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
# # check `getattr` and try it for compatibility
# def has_mps() -> bool:
#     """_summary_

#     Returns:
#         bool: _description_
#     """
#     if not getattr(torch, "has_mps", False):
#         return False
#     try:
#         torch.zeros(1).to(torch.device("mps"))
#         return True
#     except Exception:
#         return False


# def has_cuda() -> torch.device:
#     """Returns device cuda if gpu setting is enabled and cuda is present"""
#     # from modules import shared
#     device_id: Optional[Union[int, None]]
#     device_id = "mps"

#     if device_id is None:
#         return torch.device("cuda")

#     cuda_device = f"cuda:{device_id}"

#     return torch.device(cuda_device)


# def get_optimal_device() -> torch.device:
#     """_summary_

#     Args:fan
#         args (argparse.Namespace): _description_

#     Returns:
#         _type_: _description_
#     """
#     if torch.cuda.is_available():
#         return has_cuda(settings)
#     return torch.device("mps") if has_mps() else cpu


# def torch_gc() -> None:
#     """_summary_"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def enable_tf32() -> None:
#     """_summary_"""
#     if torch.cuda.is_available():
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True


# errors.run(enable_tf32, "Enabling TF32")

# # sourcery skip: avoid-global-varibles
# cpu = torch.device("cpu")
# device = (
#     device_interrogate
# ) = (
#     device_gfpgan
# ) = device_swinir = device_esrgan = device_scunet = device_codeformer = None
# dtype = torch.float16
# dtype_vae = torch.float16


# # MPS workaround for https://github.com/pytorch/pytorch/issues/79383
# def mps_contiguous(input_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
#     """Returns a contiguous in memory tensor containing the same data as self tensor. If self tensor is already in the specified memory format, this function returns the self tensor.

#     Args:
#         input_tensor (torch.Tensor): _description_
#         device (torch.device): _description_

#     Returns:
#         _type_: _description_
#     """
#     return input_tensor.contiguous() if device.type == "mps" else input_tensor


# def mps_contiguous_to(input_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
#     """_summary_

#     Args:
#         input_tensor (torch.Tensor): _description_
#         device (torch.device): _description_

#     Returns:
#         _type_: _description_
#     """
#     return mps_contiguous(input_tensor, device).to(device)


# def mps_check() -> None:
#     """_summary_"""
#     # Check that MPS is available
#     if torch.backends.mps.is_available():
#         # ic(torch.has_mps)
#         if torch.backends.mps.is_available():
#             mps_device = torch.device("mps")
#             x_ones = torch.ones(1, device=mps_device)
#             print(x_ones)
#         else:
#             print("MPS device not found.")

#         mps_device = torch.device("mps")

#         # Create a Tensor directly on the mps device
#         x_ones = torch.ones(5, device=mps_device)
#         # Or
#         x_ones = torch.ones(5, device="mps")

#         # Any operation happens on the GPU
#         x_ones * 2  # pylint: disable=pointless-statement

#     elif not torch.backends.mps.is_built():
#         print(
#             "MPS not available because the current PyTorch install was not "
#             "built with MPS enabled.",
#         )
#     else:
#         print(
#             "MPS not available because the current MacOS version is not 12.3+ "
#             "and/or you do not have an MPS-enabled device on this machine.",
#         )


# # SOURCE: https://github.com/pytorch/pytorch/issues/77988
# def seed_everything(seed: int) -> None:
#     """_summary_

#     Args:
#         seed (int): _description_
#     """
#     # Ref: https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
#     # pylint: disable=import-outside-toplevel
#     import os
#     import random

#     import numpy as np
#     import torch  # pylint: disable=reimported

#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True
