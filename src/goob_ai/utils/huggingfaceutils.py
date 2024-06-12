# from __future__ import annotations

# import torch
# import torch.nn as nn

# from huggingface_hub import PyTorchModelHubMixin


# # SOURCE: https://huggingface.co/docs/hub/en/models-uploading#upload-a-pytorch-model-using-huggingfacehub
# def upload_to_huggingface(
#     model: nn.Module, model_name: str, model_description: str, model_tags: list[str], model_dir: str
# ) -> None:
#     """
#     Summary:
#     Upload a model to the Hugging Face model hub.

#     Args:
#     model (nn.Module): The model to upload.
#     model_name (str): The name of the model.
#     model_description (str): A description of the model.
#     model_tags (list[str]): A list of tags for the model.
#     model_dir (str): The directory in which to save the model.
#     """
#     model.save_pretrained(model_dir)
#     hub_model = PyTorchModelHubMixin.from_pretrained(model_dir)
#     hub_model.push_to_hub(model_name, model_description, model_tags)
#     print(f"Model uploaded to Hugging Face model hub as {model_name}.")

#     # reload
#     model = MyModel.from_pretrained("username/my-awesome-model")
