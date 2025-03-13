from typing import Any, Callable, List, Dict, Optional, Tuple
import torch
from torchvision.datasets import VisionDataset
import albumentations as A
import numpy as np
from PIL import Image
import os

from .utils import DetectionOutput
from ...data_converter.detection import _convert_bbox_centerxywh_to_xyxy

class YoloDetection(VisionDataset, DetectionOutput):
    """
    Dataset from YOLO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    ann_dir : str
        Path to annotation text file folder
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    """
    def __init__(
        self, 
        root: str, 
        ann_dir: str,
        idx_to_class: Dict[int, str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.idx_to_class = idx_to_class
        self.ids = os.listdir(root)
        self.images = [os.path.join(root, image_id) for image_id in self.ids]
        self.targets = [os.path.join(ann_dir, image_id.replace('png', 'txt').replace('jpg', 'txt')) for image_id in self.ids]
        assert len(self.images) == len(self.targets)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images[index]).convert("RGB")

    # def _load_target(self, index: int, w: int, h: int) -> List[Any]:
    #     ann_path = self.targets[index]
    #     # If annotation text file doesn't exist
    #     if not os.path.exists(ann_path):
    #         boxes = torch.zeros(size=(0, 4))
    #         labels = torch.tensor([])
    #     else:  # If annotation text file exists
    #         # Read text file 
    #         with open(ann_path) as f:
    #             lines = f.readlines()
    #         # Get the labels
    #         labels = [int(line.split(' ')[0]) for line in lines]
    #         labels = torch.tensor(labels)
    #         # Get the bounding boxes
    #         rect_list = [line.split(' ')[1:] for line in lines]
    #         rect_list = [[float(cell.replace('\n','')) for cell in rect] for rect in rect_list]
    #         boxes = [_convert_bbox_centerxywh_to_xyxy(*rect) for rect in rect_list]  # Convert [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
    #         boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=float)  # Convert normalized coordinates to raw coordinates
    #     target = {'boxes': boxes, 'labels': labels, 'image_path': self.images[index]}
    #     return target
    
    def _load_target(self, index: int) -> List[Any]:
        ann_path = self.targets[index]
        # If annotation text file doesn't exist
        if not os.path.exists(ann_path):
            labels = []
            boxes = []            
        else:  # If annotation text file exists
            # Read text file 
            with open(ann_path) as f:
                lines = f.readlines()
            # Get the labels
            labels = [int(line.split(' ')[0]) for line in lines]
            # Get the bounding boxes
            rect_list = [line.split(' ')[1:] for line in lines]
            boxes = [[float(cell.replace('\n','')) for cell in rect] for rect in rect_list]
        return labels, boxes
    
    def _convert_target(self, boxes, labels, index: int, w: int, h: int):
        """Convert COCO to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64)
        # Convert the bounding boxes from [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
        boxes = [[int(k) for k in _convert_bbox_centerxywh_to_xyxy(*box)]
                  for box in boxes]
        # Convert normalized coordinates to raw coordinates
        boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=torch.float32) if len(boxes) > 0 \
                else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Get the image path
        target = {'boxes': boxes, 'labels': labels, 'image_path': self.images[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        labels, boxes = self._load_target(id)
        w, h = image.size

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), bboxes=boxes, class_labels=labels)
                image = transformed['image']
                target = self._convert_target(transformed['bboxes'], 
                                              transformed['class_labels'], index, w, h)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, index, w, h)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, id)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)
