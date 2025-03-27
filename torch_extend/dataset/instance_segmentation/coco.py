from typing import Any, Callable, List, Optional, Tuple, Literal
import torch
import torchvision.datasets as ds
from transformers import BaseImageProcessor
import albumentations as A
import numpy as np
import os

from ..detection.coco import CocoDetection
from ...data_converter.semantic_segmentation import convert_polygon_to_mask
from ...data_converter.detection import _convert_bbox_xywh_to_xyxy
from ...data_converter.instance_segmentation import convert_image_target_to_transformers

class CocoInstanceSegmentation(CocoDetection):
    """
    COCO format Instance Segmentation Dataset

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
    out_fmt : Literal["torchvision", "transformers"]
        The output format of the image and target, either ``"torchvision"`` or ``"transformers"``.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    processor : callable, optional
        An image processor instance for HuggingFace Transformers. Only available if ``out_fmt="transformers"``.
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        processor: Optional[BaseImageProcessor] = None
    ) -> None:
        super().__init__(root, annFile, out_fmt, transform, target_transform, transforms, False, processor)
        self.border_idx = None  # Border index is not used in COCO format
        # Background index is 0 in default
        self.bg_idx = 0
        if out_fmt == "transformers":
            self.border_idx = None  # Border index is not used in Transformers format
            # If `do_reduce_labels=True` in the processor, `idx_to_class` is also reduced and background index is set to `processor.ignore_index`
            if self.processor.do_reduce_labels:
                self.idx_to_class = {k-1: v for k, v in self.idx_to_class.items()}
                self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
                self.bg_idx = self.processor.ignore_index

    def _load_target(self, id: int, height: int, width: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        # Get the segmentation polygons
        segmentations = [obj['segmentation'] for obj in target_src]
        # Convert the polygons to masks
        masks = convert_polygon_to_mask(segmentations, height, width)
        masks = np.stack(masks, axis=0) if len(masks) > 0 else np.zeros((0, height, width), dtype=np.uint8)
        # Get the bounding boxes
        boxes = []
        for obj, mask in zip(target_src, masks):
            # If the bounding box exists
            if len(obj['bbox']) > 0:
                boxes.append(obj['bbox'])
            # If the bounding box does not exist, create a bounding box from the mask
            else:
                nonzero_mask = mask.nonzero()
                boxes.append([nonzero_mask[:, 1].min(), nonzero_mask[:, 0].min(),
                            nonzero_mask[:, 1].max()-nonzero_mask[:, 1].min(),
                            nonzero_mask[:, 0].max()-nonzero_mask[:, 0].min()])
        return boxes, labels, masks
    
    def _convert_target(self, boxes, labels, masks, id, h, w):
        """Convert COCO to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(size=(0,), dtype=torch.int64)
        # Convert the bounding boxes
        boxes = [[int(k) for k in _convert_bbox_xywh_to_xyxy(*box)]
                  for box in boxes]
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Convert the instance masks
        masks = torch.stack([mask if isinstance(mask, torch.Tensor) else torch.tensor(mask, dtype=torch.uint8) for mask in masks]) if len(masks) > 0 else torch.zeros(size=(0, h, w), dtype=torch.uint8)
        # Miscellaneous fields
        image_id = id
        area = torch.tensor([mask.sum() for mask in masks], dtype=torch.float32)  # Mask areas
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)  # suppose all instances are not crowd
        # Get the image path
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        target = {'boxes': boxes,
                  'labels': labels,
                  'masks': masks,
                  'image_id': image_id,
                  'area': area,
                  'iscrowd': iscrowd,
                  'image_path': os.path.join(self.root, image_path)}
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        w, h = image.size
        boxes, labels, masks = self._load_target(id, h, w)
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
                                              id, h, w)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, masks, id, h, w)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, masks, id, h, w)

        # Output as TorchVision format
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format
        # If the output image sizes are different, `processor.pad(pixel_values, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, index, self.processor, out_fmt='detr')
