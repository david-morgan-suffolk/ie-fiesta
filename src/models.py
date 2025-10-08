from __future__ import annotations
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_frcnn_mobilenet(num_classes: int, elongated_anchors: bool = True):
    """
    Faster R-CNN with MobileNetV3-FPN backbone.
    num_classes: includes background → if you have 2 categories, pass 3.
    """
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if elongated_anchors:
        model.rpn.anchor_generator.sizes = ((32, 64, 128, 256, 512),)
        model.rpn.anchor_generator.aspect_ratios = ((0.5, 1.0, 1.5, 2.0, 3.0),)

    model.roi_heads.nms_thresh = 0.6
    return model
