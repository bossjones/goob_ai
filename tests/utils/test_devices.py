from __future__ import annotations

import torch

from goob_ai.utils.devices import get_device

import pytest


@pytest.mark.parametrize(
    "mps_available, cuda_available, expected_device",
    [
        (True, True, torch.device("mps")),  # MPS available
        (True, False, torch.device("mps")),  # MPS available, CUDA not available
        (False, True, torch.device("cuda")),  # MPS not available, CUDA available
        (False, False, torch.device("cpu")),  # Neither MPS nor CUDA available
    ],
    ids=[
        "MPS available",
        "MPS available, CUDA not available",
        "MPS not available, CUDA available",
        "Neither MPS nor CUDA available",
    ],
)
def test_get_device(mps_available, cuda_available, expected_device, mocker):
    # Arrange
    mocker.patch("torch.backends.mps.is_available", return_value=mps_available)
    mocker.patch("torch.cuda.is_available", return_value=cuda_available)

    # Act
    device = get_device()

    # Assert
    assert device == expected_device
