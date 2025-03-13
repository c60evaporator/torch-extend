from typing import Any, Callable, List, Dict, Optional, Tuple, Literal
import torch
import albumentations as A
import numpy as np
from PIL import Image
import os
from transformers import BaseImageProcessor

import xml.etree.ElementTree as ET

from ..detection.voc import VOCDetection
from ..detection.utils import DetectionOutput

class VOCInstanceSegmentation(VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Instance Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    output_format : Literal["torchvision", "transformers"]
        The output format of the image and target, either ``"torchvision"`` or ``"transformers"``.
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    download : bool, optional
        If true, downloads VOC2012 dataset from the internet and puts it in root directory.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    processor : callable, optional
        An image processor instance for HuggingFace Transformers. Only available if ``output_format="transformers"``. 
    border_idx : int
        The index of the border in the target mask.
    """

    def __init__(
        self,
        root: str,
        idx_to_class: Dict[int, str] = None,
        output_format: Literal["torchvision", "transformers"] = "torchvision",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        processor: Optional[BaseImageProcessor] = None,
        border_idx: int = 255
    ):
        super().__init__(root, idx_to_class, output_format, image_set, download,
                         transform, target_transform, transforms, processor)
        self.border_idx = border_idx

    def __len__(self) -> int:
        return len(self.images_instance)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_instance[index]).convert("RGB")
    
    def _load_target(self, index: int) -> List[Any]:
        """Load the target masks and bboxes of the dataset"""
        # Read bounding box XML file 
        objects = self._load_voc_bboxes(self.bboxes_instance[index])
        # Get the labels
        labels = [self.class_to_idx[obj['name']] for obj in objects]
        # Get the bounding boxes
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        # Read the mask
        mask = Image.open(self.masks_instance[index])
        mask = np.array(mask, dtype=np.uint8)
        # Instance validation
        instance_ids = np.arange(1, len(labels)+1)
        mask_instances = np.unique(mask)[1:]  # Remove background (0)
        mask_instances = mask_instances[mask_instances != self.border_idx]  # Remove border
        if not np.array_equal(mask_instances, instance_ids):
            print(f'Warning: Bounding box with empty mask found in image{index}, creating dummy mask.')
        # Split the mask into instance masks
        masks = [(mask == instance_id).astype(np.uint8) for instance_id in instance_ids]
        # Border mask
        border_mask = (mask == self.border_idx).astype(np.uint8)
        return boxes, labels, masks, border_mask
    
    def _convert_target(self, boxes, labels, masks, border_mask, index, w, h):
        """Convert VOC to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(boxes) > 0 else torch.zeros(size=(0,), dtype=torch.float32)
        # Convert the bounding boxes
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Convert the instance masks
        masks = torch.stack([mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.uint8) for mask in masks]) if len(masks) > 0 else torch.zeros(size=(0, h, w), dtype=torch.uint8)
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
                  'image_path': self.images_instance[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        image = self._load_image(index)
        boxes, labels, masks, border_mask = self._load_target(index)
        w, h = image.size
        if len(masks) == 0:  # masks cannot be empty in Albumentations, so we add a dummy mask
            print('Warning: Empty masks found, creating dummy masks')
            masks = [np.zeros((h, w), dtype=np.uint8)]

        # Apply transforms
        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), bboxes=boxes, class_labels=labels, masks=masks)
                image = transformed['image']
                target = self._convert_target(transformed['bboxes'], 
                                            transformed['class_labels'],
                                            transformed['masks'],
                                            border_mask, index, w, h)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, masks, border_mask, index, w, h)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, masks, border_mask, index, w, h)
        
        # Output as TorchVision format
        if self.output_format == "torchvision":
            return image, target

        # Output as Transformers format
        elif self.output_format == "transformers":
            return self._convert_to_transformers(index, image, target)
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_instance[index], self.masks_instance[index]
