from typing import Any, Callable, List, Dict, Optional, Tuple, Literal
import torch
import albumentations as A
from torchvision.transforms import v2
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
    out_fmt : Literal["torchvision", "transformers"]
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
    reduce_labels : bool
        If True, the label 0 is regarded as the background and all the labels will be reduced by 1. Also, the class_to_idx will

        For example, if the labels are [1, 3] and the class_to_idx is {1: 'aeroplane', 2: 'bicycle', 3: 'bird'}, the labels will be [0, 2] and the class_to_idx will be {0: 'aeroplane', 1: 'bicycle', 2: 'bird'}.
    processor : callable, optional
        An image processor instance for HuggingFace Transformers. Only available if ``out_fmt="transformers"``. 
    border_idx : int
        The index of the border in the target mask.
    """

    def __init__(
        self,
        root: str,
        idx_to_class: Dict[int, str] = None,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        reduce_labels: bool = False,
        processor: Optional[BaseImageProcessor] = None,
        border_idx: int = 255,
    ):
        super().__init__(root, idx_to_class, out_fmt, image_set, download,
                         transform, target_transform, transforms,
                         reduce_labels, processor)
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
        labels = torch.tensor(labels, dtype=torch.int64) if len(boxes) > 0 else torch.zeros(size=(0,), dtype=torch.int64)
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
    
    def _convert_target_to_transformers(self, image, image_id, target):
        """
        Convert VOC to Transformers format

        Returns
        -------
        {"pixel_values": torch.Tensor(C, H, W), "pixel_mask": torch.Tensor(H, W), "mask_labels": torch.Tensor(n_instances, H, W), "class_labels": torch.Tensor(n_instances)}

        If the output image sizes are different, `processor.encode_inputs(pixel_values, segmentation_map, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        """
        # Convert instance masks into one mask with instance IDs
        segmentation_map = torch.zeros_like(target['masks'][0])
        for i, mask in enumerate(target['masks']):
            segmentation_map[mask.bool()] = i+1
        # mapping between instance IDs and class labels
        inst2class = {i: label for i, label in enumerate([0] + target['labels'].tolist())}
        # Apply the Transformers processor
        encoding = self.processor(images=image, segmentation_maps=segmentation_map,
                                  instance_id_to_semantic_id=inst2class,
                                  return_tensors="pt")
        # If images sizes are the same
        if self.same_img_size:
            # Remove batch dimension and return as dictionary
            return {
                "pixel_values": encoding["pixel_values"].squeeze(),
                "pixel_mask": encoding["pixel_mask"].squeeze(),
                "mask_labels": encoding["mask_labels"][0][1:, :, :],  # remove background
                "class_labels": encoding["class_labels"][0][1:]  # remove background
            }
        # If images sizes are different, `processor.encode_inputs(pixel_values, segmentation_map, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        else:
            # Recreate the segmentation map
            segmentation_map = torch.zeros_like(encoding["pixel_mask"].squeeze())
            for i, mask in enumerate(encoding["mask_labels"][0][1:, :, :]):
                segmentation_map[mask.bool()] = i+1
            # Remove batch dimension and return as dictionary
            return {
                "images": encoding["pixel_values"].squeeze(),
                "segmentation_maps": segmentation_map,
                "instance_id_to_semantic_id": inst2class
            }

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
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format
        elif self.out_fmt == "transformers":
            return self._convert_target_to_transformers(image, index, target)
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_instance[index], self.masks_instance[index]
