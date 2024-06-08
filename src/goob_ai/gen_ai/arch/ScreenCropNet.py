"""ScreenCropNet"""

from __future__ import annotations

import timm
import torch.nn as nn
import torchvision.models as models

from loguru import logger as LOGGER


MODEL_NAMES = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


####################################################################
# SOURCE: https://github.com/mukkaragayathri23/FusedMammoNet/blob/bb6e436818335bc73b1c2b57a725e7f5515faf89/models.py#L33
def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def unfreeze_last_layer(model, last_layer_name="classifier"):
    if last_layer_name.lower() == "classifier":
        for param in model.classifier.parameters():
            param.requires_grad = True

    if last_layer_name.lower() == "fc":
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


####################################################################


class ObjLocModel(nn.Module):
    def __init__(self, num_classes=4):
        super(ObjLocModel, self).__init__()
        # self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4)
        # self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4)
        # num_classes=4

        # SOURCE: https://github.com/AntoonGa/VisionTransformerFromScratch/blob/d19e5562bcbfa30c3480d60fbd9a703b4841efb9/core/classifier_models/efficientnet_b0.py#L12
        # Where we define all the parts of the model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        # in_features = self.base_model.classifier.in_features
        # self.base_model.classifier = nn.Linear(in_features, num_classes)

        # NOTE: trying this now = https://github.com/mehmetcanbudak/PyTorch_Card_Classify/blob/d839014617b1a9ad968db9289f40dfd225085326/model.py#L83
        enet_out_size = 1280
        # Make a classifier
        self.base_model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(enet_out_size, num_classes))

        # self.setup_model()

    # SOURCE: https://github.com/TeamEpochGithub/iv-q3-harmful-brain-activity/blob/cc3acae5220ca6f073578560a0f1353cb7871923/src/modules/training/models/torchvision.py#L9
    # def setup_model(self) -> None:
    #     """Set up the first layer and last_layer based on the models architecture."""
    #     match self.model.__class__.__name__:
    #         case "EfficientNet":
    #             first = self.model.features[0][0]
    #             new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
    #             self.model.features[0][0] = new_layer
    #             num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
    #             self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer
    #         case "VGG":
    #             first = self.model.features[0]
    #             new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
    #             self.model.features[0] = new_layer
    #             num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
    #             self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer
    #         case "ResNet":
    #             first = self.model.conv1
    #             new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
    #             self.model.conv1 = new_layer
    #             # Replace the last layer
    #             num_features = self.model.fc.in_features
    #             self.model.fc = nn.Linear(num_features, self.out_channels)
    #         case _:
    #             logger.warning("Model not fully implemented yet.. Might crash, reverting to baseline implementation.")
    #             first = self.model.features[0]
    #             new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
    #             self.model.features[0] = new_layer
    #             num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
    #             self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer

    def forward(self, images, gt_bboxes=None):
        # bboxes_logits = self.backbone(images)  ## predicted bounding boxes
        bboxes_logits = self.base_model(images)  ## predicted bounding boxes

        # gt_bboxes = ground truth bounding boxes
        if gt_bboxes != None:
            # Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.
            loss = nn.MSELoss()(bboxes_logits, gt_bboxes)
            return bboxes_logits, loss

        return bboxes_logits
