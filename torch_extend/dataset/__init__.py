from .classification.cifar import CIFAR10TV
from .detection.voc import VOCDetection
from .detection.coco import CocoDetection
from .detection.yolo import YoloDetection
from .semantic_segmentation.voc import VOCSemanticSegmentation
from .semantic_segmentation.coco import CocoSemanticSegmentation
from .instance_segmentation.voc import VOCInstanceSegmentation
from .instance_segmentation.coco import CocoInstanceSegmentation

__all__ = [
    "CIFAR10TV",
    "VOCDetection",
    "CocoDetection",
    "YoloDetection",
    "VOCSemanticSegmentation",
    "CocoSemanticSegmentation",
    "VOCInstanceSegmentation",
    "CocoInstanceSegmentation"
]
