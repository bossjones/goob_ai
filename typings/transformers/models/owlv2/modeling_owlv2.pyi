"""
This type stub file was generated by pyright.
"""

import torch
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor, nn
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, is_vision_available, replace_return_docstrings
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig

""" PyTorch OWLv2 model."""
if is_vision_available():
    ...
logger = ...
_CHECKPOINT_FOR_DOC = ...
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    ...

def owlv2_loss(similarity: torch.Tensor) -> torch.Tensor:
    ...

@dataclass
class Owlv2Output(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size * num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of
            [`Owlv2VisionModel`].
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    logits_per_image: torch.FloatTensor = ...
    logits_per_text: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    ...

def box_iou(boxes1, boxes2): # -> tuple[Any, Any]:
    ...

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    ...

@dataclass
class Owlv2ObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        objectness_logits (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`):
            The objectness logits of all image patches. OWL-ViT represents images as a set of image patches where the
            total number of patches is (image_size / patch_size)**2.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, num_max_text_queries, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`Owlv2TextModel`].
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes image
            embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """
    loss: Optional[torch.FloatTensor] = ...
    loss_dict: Optional[Dict] = ...
    logits: torch.FloatTensor = ...
    objectness_logits: torch.FloatTensor = ...
    pred_boxes: torch.FloatTensor = ...
    text_embeds: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    class_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


@dataclass
class Owlv2ImageGuidedObjectDetectionOutput(ModelOutput):
    """
    Output type of [`Owlv2ForObjectDetection.image_guided_detection`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, num_patches, num_queries)`):
            Classification logits (including no-object) for all queries.
        target_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual target image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        query_pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_patches, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual query image in the batch
            (disregarding possible padding). You can use [`~Owlv2ImageProcessor.post_process_object_detection`] to
            retrieve the unnormalized bounding boxes.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        query_image_embeds (`torch.FloatTensor` of shape `(batch_size, patch_size, patch_size, output_dim`):
            Pooled output of [`Owlv2VisionModel`]. OWLv2 represents images as a set of image patches and computes
            image embeddings for each patch.
        class_embeds (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
            Class embeddings of all image patches. OWLv2 represents images as a set of image patches where the total
            number of patches is (image_size / patch_size)**2.
        text_model_output (Tuple[`BaseModelOutputWithPooling`]):
            The output of the [`Owlv2TextModel`].
        vision_model_output (`BaseModelOutputWithPooling`):
            The output of the [`Owlv2VisionModel`].
    """
    logits: torch.FloatTensor = ...
    image_embeds: torch.FloatTensor = ...
    query_image_embeds: torch.FloatTensor = ...
    target_pred_boxes: torch.FloatTensor = ...
    query_pred_boxes: torch.FloatTensor = ...
    class_embeds: torch.FloatTensor = ...
    text_model_output: BaseModelOutputWithPooling = ...
    vision_model_output: BaseModelOutputWithPooling = ...
    def to_tuple(self) -> Tuple[Any]:
        ...
    


class Owlv2VisionEmbeddings(nn.Module):
    def __init__(self, config: Owlv2VisionConfig) -> None:
        ...
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        ...
    


class Owlv2TextEmbeddings(nn.Module):
    def __init__(self, config: Owlv2TextConfig) -> None:
        ...
    
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., inputs_embeds: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...
    


class Owlv2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ...) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        ...
    


class Owlv2MLP(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
    


class Owlv2EncoderLayer(nn.Module):
    def __init__(self, config: Owlv2Config) -> None:
        ...
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        ...
    


class Owlv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = Owlv2Config
    base_model_prefix = ...
    supports_gradient_checkpointing = ...
    _no_split_modules = ...


OWLV2_START_DOCSTRING = ...
OWLV2_TEXT_INPUTS_DOCSTRING = ...
OWLV2_VISION_INPUTS_DOCSTRING = ...
OWLV2_INPUTS_DOCSTRING = ...
OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING = ...
OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING = ...
class Owlv2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Owlv2EncoderLayer`].

    Args:
        config: Owlv2Config
    """
    def __init__(self, config: Owlv2Config) -> None:
        ...
    
    def forward(self, inputs_embeds, attention_mask: Optional[torch.Tensor] = ..., causal_attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`).
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        ...
    


class Owlv2TextTransformer(nn.Module):
    def __init__(self, config: Owlv2TextConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2TextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., position_ids: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        ...
    


class Owlv2TextModel(Owlv2PreTrainedModel):
    config_class = Owlv2TextConfig
    def __init__(self, config: Owlv2TextConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    def set_input_embeddings(self, value): # -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2TextConfig)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, Owlv2TextModel

        >>> model = Owlv2TextModel.from_pretrained("google/owlv2-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        ...
    


class Owlv2VisionTransformer(nn.Module):
    def __init__(self, config: Owlv2VisionConfig) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfig)
    def forward(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        ...
    


class Owlv2VisionModel(Owlv2PreTrainedModel):
    config_class = Owlv2VisionConfig
    main_input_name = ...
    def __init__(self, config: Owlv2VisionConfig) -> None:
        ...
    
    def get_input_embeddings(self) -> nn.Module:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Owlv2VisionConfig)
    def forward(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Owlv2VisionModel

        >>> model = Owlv2VisionModel.from_pretrained("google/owlv2-base-patch16")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        ...
    


@add_start_docstrings(OWLV2_START_DOCSTRING)
class Owlv2Model(Owlv2PreTrainedModel):
    config_class = Owlv2Config
    def __init__(self, config: Owlv2Config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor] = ..., attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`Owlv2TextModel`].

        Examples:
        ```python
        >>> from transformers import AutoProcessor, Owlv2Model

        >>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> inputs = processor(
        ...     text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt"
        ... )
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`Owlv2VisionModel`].

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Owlv2Model

        >>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2Output, config_class=Owlv2Config)
    def forward(self, input_ids: Optional[torch.LongTensor] = ..., pixel_values: Optional[torch.FloatTensor] = ..., attention_mask: Optional[torch.Tensor] = ..., return_loss: Optional[bool] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_base_image_embeds: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Union[Tuple, Owlv2Output]:
        r"""
        Returns:

        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Owlv2Model

        >>> model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        ...
    


class Owlv2BoxPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config, out_dim: int = ...) -> None:
        ...
    
    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        ...
    


class Owlv2ClassPredictionHead(nn.Module):
    def __init__(self, config: Owlv2Config) -> None:
        ...
    
    def forward(self, image_embeds: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor], query_mask: Optional[torch.Tensor]) -> Tuple[torch.FloatTensor]:
        ...
    


class Owlv2ForObjectDetection(Owlv2PreTrainedModel):
    config_class = Owlv2Config
    def __init__(self, config: Owlv2Config) -> None:
        ...
    
    @staticmethod
    def normalize_grid_corner_coordinates(num_patches: int) -> torch.Tensor:
        ...
    
    def objectness_predictor(self, image_features: torch.FloatTensor) -> torch.FloatTensor:
        """Predicts the probability that each image feature token is an object.

        Args:
            image_features (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_dim)`)):
                Features extracted from the image.
        Returns:
            Objectness scores.
        """
        ...
    
    @lru_cache(maxsize=2)
    def compute_box_bias(self, num_patches: int, feature_map: Optional[torch.FloatTensor] = ...) -> torch.Tensor:
        ...
    
    def box_predictor(self, image_feats: torch.FloatTensor, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        ...
    
    def class_predictor(self, image_feats: torch.FloatTensor, query_embeds: Optional[torch.FloatTensor] = ..., query_mask: Optional[torch.Tensor] = ...) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        ...
    
    def image_text_embedder(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: torch.Tensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        ...
    
    def image_embedder(self, pixel_values: torch.FloatTensor, output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ...) -> Tuple[torch.FloatTensor]:
        ...
    
    def embed_image_query(self, query_image_features: torch.FloatTensor, query_feature_map: torch.FloatTensor) -> torch.FloatTensor:
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_IMAGE_GUIDED_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ImageGuidedObjectDetectionOutput, config_class=Owlv2Config)
    def image_guided_detection(self, pixel_values: torch.FloatTensor, query_pixel_values: Optional[torch.FloatTensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Owlv2ImageGuidedObjectDetectionOutput:
        r"""
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import torch
        >>> import numpy as np
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
        >>> query_image = Image.open(requests.get(query_url, stream=True).raw)
        >>> inputs = processor(images=image, query_images=query_image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model.image_guided_detection(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])

        >>> # Convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> results = processor.post_process_image_guided_detection(
        ...     outputs=outputs, threshold=0.9, nms_threshold=0.3, target_sizes=target_sizes
        ... )
        >>> i = 0  # Retrieve predictions for the first image
        >>> boxes, scores = results[i]["boxes"], results[i]["scores"]
        >>> for box, score in zip(boxes, scores):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")
        Detected similar object with confidence 0.938 at location [490.96, 109.89, 821.09, 536.11]
        Detected similar object with confidence 0.959 at location [8.67, 721.29, 928.68, 732.78]
        Detected similar object with confidence 0.902 at location [4.27, 720.02, 941.45, 761.59]
        Detected similar object with confidence 0.985 at location [265.46, -58.9, 1009.04, 365.66]
        Detected similar object with confidence 1.0 at location [9.79, 28.69, 937.31, 941.64]
        Detected similar object with confidence 0.998 at location [869.97, 58.28, 923.23, 978.1]
        Detected similar object with confidence 0.985 at location [309.23, 21.07, 371.61, 932.02]
        Detected similar object with confidence 0.947 at location [27.93, 859.45, 969.75, 915.44]
        Detected similar object with confidence 0.996 at location [785.82, 41.38, 880.26, 966.37]
        Detected similar object with confidence 0.998 at location [5.08, 721.17, 925.93, 998.41]
        Detected similar object with confidence 0.969 at location [6.7, 898.1, 921.75, 949.51]
        Detected similar object with confidence 0.966 at location [47.16, 927.29, 981.99, 942.14]
        Detected similar object with confidence 0.924 at location [46.4, 936.13, 953.02, 950.78]
        ```"""
        ...
    
    @add_start_docstrings_to_model_forward(OWLV2_OBJECT_DETECTION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Owlv2ObjectDetectionOutput, config_class=Owlv2Config)
    def forward(self, input_ids: torch.Tensor, pixel_values: torch.FloatTensor, attention_mask: Optional[torch.Tensor] = ..., output_attentions: Optional[bool] = ..., output_hidden_states: Optional[bool] = ..., return_dict: Optional[bool] = ...) -> Owlv2ObjectDetectionOutput:
        r"""
        Returns:

        Examples:
        ```python
        >>> import requests
        >>> from PIL import Image
        >>> import numpy as np
        >>> import torch
        >>> from transformers import AutoProcessor, Owlv2ForObjectDetection
        >>> from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        >>> processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16-ensemble")
        >>> model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = [["a photo of a cat", "a photo of a dog"]]
        >>> inputs = processor(text=texts, images=image, return_tensors="pt")

        >>> # forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Note: boxes need to be visualized on the padded, unnormalized image
        >>> # hence we'll set the target image sizes (height, width) based on that

        >>> def get_preprocessed_image(pixel_values):
        ...     pixel_values = pixel_values.squeeze().numpy()
        ...     unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
        ...     unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
        ...     unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
        ...     unnormalized_image = Image.fromarray(unnormalized_image)
        ...     return unnormalized_image

        >>> unnormalized_image = get_preprocessed_image(inputs.pixel_values)

        >>> target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        >>> # Convert outputs (bounding boxes and class logits) to final bounding boxes and scores
        >>> results = processor.post_process_object_detection(
        ...     outputs=outputs, threshold=0.2, target_sizes=target_sizes
        ... )

        >>> i = 0  # Retrieve predictions for the first image for the corresponding text queries
        >>> text = texts[i]
        >>> boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

        >>> for box, score, label in zip(boxes, scores, labels):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")
        Detected a photo of a cat with confidence 0.614 at location [512.5, 35.08, 963.48, 557.02]
        Detected a photo of a cat with confidence 0.665 at location [10.13, 77.94, 489.93, 709.69]
        ```"""
        ...
    


