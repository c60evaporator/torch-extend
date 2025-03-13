from .classification.cifar import CIFAR10TV
from .detection.voc import VOCDetection
from .detection.coco import CocoDetectionTV
from .detection.yolo import YoloDetection
from .semantic_segmentation.voc import VOCSemanticSegmentation
from .semantic_segmentation.coco import CocoSemanticSegmentation
from .instance_segmentation.voc import VOCInstanceSegmentation

__all__ = [
    "CIFAR10TV",
    "VOCDetection",
    "CocoDetectionTV",
    "YoloDetection",
    "VOCSemanticSegmentation",
    "CocoSemanticSegmentation",
    "VOCInstanceSegmentation"
]
