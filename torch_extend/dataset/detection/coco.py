from typing import Any, Callable, List, Dict, Optional, Tuple
import torch
from torchvision.datasets import CocoDetection
import albumentations as A
import numpy as np
import os

from ...data_converter.detection import DetectionOutput, convert_bbox_xywh_to_xyxy

class CocoDetectionTV(CocoDetection, DetectionOutput):
    """
    Dataset from COCO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
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
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.class_to_idx = {
            v['name']: v['id']
            for k, v in self.coco.cats.items()
        }
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(self.class_to_idx.values())):
            if i not in self.class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'

    def _load_target(self, id: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        # Get the bounding boxes
        boxes = [obj['bbox'] for obj in target_src]
        return labels, boxes
    
    def _convert_target(self, boxes, labels, id):
        """Convert COCO to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64)
        # Convert the bounding boxes
        boxes = [[int(k) for k in convert_bbox_xywh_to_xyxy(*box)]
                  for box in boxes]
        boxes = torch.tensor(boxes, dtype=torch.int64) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.int64)
        # Get the image path
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        target = {'boxes': boxes, 'labels': labels, 'image_path': os.path.join(self.root, image_path)}
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        labels, boxes = self._load_target(id)

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), bboxes=boxes, class_labels=labels)
                image = transformed['image']
                target = self._convert_target(transformed['bboxes'], 
                                              transformed['class_labels'], id)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, id)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, id)

        return image, target
