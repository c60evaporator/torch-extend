from typing import Any, Callable, List, Dict, Optional, Tuple
import torch
from torchvision import tv_tensors
import albumentations as A
import numpy as np
from PIL import Image
import os

import xml.etree.ElementTree as ET

from ..detection.voc import VOCBaseTV, parse_voc_xml
from ..detection.utils import DetectionOutput

class VOCInstanceSegmentation(VOCBaseTV, DetectionOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Instance Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    border_idx : int
        The index of the border in the target mask.
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    download : bool, optional
        If true, downloads VOC2012 dataset from the internet and puts it in root directory.
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
        idx_to_class: Dict[int, str] = None,
        border_idx: int = 255,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, download, transform, target_transform, transforms)
        if idx_to_class is None:
            self.idx_to_class = self.IDX_TO_CLASS
        else:
            self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.ids = os.listdir(root)
        self.border_idx = border_idx

    def __len__(self) -> int:
        return len(self.images_instance)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_instance[index]).convert("RGB")
    
    def _load_target(self, index: int) -> List[Any]:
        """Load the target masks and bboxes of the dataset"""
        # Read XML file 
        target = parse_voc_xml(ET.parse(self.bboxes_instance[index]).getroot())
        objects = target['annotation']['object']
        # Get the labels
        labels = [self.class_to_idx[obj['name']] for obj in objects]
        # Get the bounding boxes
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        # Read the mask
        mask = Image.open(self.masks_semantic[index])
        mask = np.array(mask, dtype=np.uint8)
        instance_ids = np.unique(mask)[1:]  # Remove background (0)
        instance_ids = instance_ids[instance_ids != self.border_idx]  # Remove border
        # Split the mask into instance masks
        masks = [(mask == instance_id).astype(np.uint8) for instance_id in instance_ids]
        # Border mask
        border_mask = (mask == self.border_idx).astype(np.uint8)
        return boxes, labels, masks, border_mask
    
    def _convert_target(self, boxes, labels, masks, border_mask, index):
        """Convert VOC to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(boxes) > 0 else torch.zeros(size=(0,), dtype=torch.float32)
        # Convert the bounding boxes
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Convert the instance masks
        masks = torch.stack([mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.uint8) for mask in masks]) if len(masks) > 0 else torch.zeros(size=(0,), dtype=torch.uint8)
        # Border mask
        border_mask = torch.tensor(border_mask, dtype=torch.uint8)
        # Miscellaneous fields
        image_id = index
        area = torch.tensor([mask.sum() for mask in masks], dtype=torch.float32)  # Mask areas
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)  # suppose all instances are not crowd
        # Get the image path
        target = {'boxes': boxes,
                  'labels': labels,
                  'masks': masks,
                  'image_id': image_id,
                  'area': area,
                  'iscrowd': iscrowd,
                  'border_mask': border_mask,
                  'image_path': self.images_detection[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        image = self._load_image(index)
        boxes, labels, masks, border_mask = self._load_target(index)

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), bboxes=boxes, class_labels=labels, masks=masks)
                image = transformed['image']
                target = self._convert_target(transformed['bboxes'], 
                                              transformed['class_labels'],
                                              transformed['masks'],
                                              border_mask, index)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, masks, border_mask, index)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, masks, border_mask, index)

        return image, target
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_instance[index], self.masks_instance[index]
