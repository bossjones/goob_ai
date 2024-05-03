"""ScreenCropNet"""

import timm
import torch.nn as nn
import torchvision.models as models

MODEL_NAMES = sorted(
    name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class ObjLocModel(nn.Module):
    def __init__(self):
        super(ObjLocModel, self).__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4)

    def forward(self, images, gt_bboxes=None):
        bboxes_logits = self.backbone(images)  ## predicted bounding boxes

        # gt_bboxes = ground truth bounding boxes
        if gt_bboxes != None:
            # Creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.
            loss = nn.MSELoss()(bboxes_logits, gt_bboxes)
            return bboxes_logits, loss

        return bboxes_logits
