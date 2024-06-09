```python
from cerebro_bot.utils.arch.ScreenCropNet import ObjLocModel as ScreenCropNet_ObjLocModel
from cerebro_bot.aio_settings import aiosettings
import torch
import pathlib
import rich
from torch import nn
from cerebro_bot import debugger

# NOTE: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
def load_model_for_inference(save_path: str, device: str) -> nn.Module:
    model = ScreenCropNet_ObjLocModel()
    model.name = "ObjLocModelV1"
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f"Model loaded from path {save_path} successfully.")
    # Get the model size in bytes then convert to megabytes
    model_size = pathlib.Path(save_path).stat().st_size // (1024 * 1024)
    print(f"{save_path} | feature extractor model size: {model_size} MB")
    return model


# wrapper function of common code
def run_get_model_for_inference(
    model: torch.nn.Module,
    device: torch.device,
    path_to_model: pathlib.PosixPath,
) -> torch.nn.Module:
    """wrapper function to load model .pth file from disk

    Args:
        model (torch.nn.Module): _description_
        device (torch.device): _description_
        class_names (List[str]): _description_

    Returns:
        Tuple[pathlib.PosixPath, torch.nn.Module]: _description_
    """
    return load_model_for_inference(path_to_model, device)

model_name: str = "ScreenCropNetV1_378_epochs.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# async def load_model(self, model_name: str = "ScreenCropNetV1_378_epochs.pth"):
model = ScreenCropNet_ObjLocModel()
model.name = "ObjLocModelV1"
model.to(device)
weights = f"{aiosettings.screencropnet_dir}/{model_name}"
model = run_get_model_for_inference(model, device, weights)
# self.bot.autocrop_model = model
print(f"Loaded model: {weights} ...")
```
