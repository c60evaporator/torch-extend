from typing import Any, Callable, List, Optional, Tuple, Literal
import torch
import torchvision.datasets as ds
from transformers import BaseImageProcessor
import albumentations as A
import numpy as np
import os

from ...data_converter.detection import _convert_bbox_xywh_to_xyxy
from ...validate.common import validate_same_img_size
from .utils import DetectionOutput
from ...data_converter.detection import convert_image_target_to_transformers

class CocoDetection(ds.CocoDetection, DetectionOutput):
    """
    Dataset from COCO format to Torchvision or Transformers format with image path

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
    reduce_labels : bool
        If True, the label 0 is regarded as the background and all the labels will be reduced by 1. Also, the class_to_idx will be updated accordingly.

        For example, if the labels are [1, 3] and the class_to_idx is {1: 'aeroplane', 2: 'bicycle', 3: 'bird'}, the labels will be [0, 2] and the class_to_idx will be {0: 'aeroplane', 1: 'bicycle', 2: 'bird'}.
    processor : callable, optional
        An image processor instance for HuggingFace Transformers. Only available if ``out_fmt="transformers"``.

    The dataset folder structure should be like this:
    ```
    root/
        ├── annotations/
        │   ├── instances_train2017.json <-- This file should be set as `annFile` if the dataset is for training
        │   ├── instances_val2017.json <-- This file should be set as `annFile` if the dataset is for validation
        │   └── ...
        ├── train2017/ <-- This folder should be set as `root` if the dataset is for training
        │   ├── 000000000001.jpg
        │   ├── 000000000002.jpg
        │   └── ...
        └── val2017/ <-- This folder should be set as `root` if the dataset is for validation
            ├── 000000010001.jpg
            ├── 000000010002.jpg
            └── ...
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        reduce_labels: bool = False,
        processor: Optional[BaseImageProcessor] = None
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        # Index to class dictionary
        self.idx_to_class = {
            v['id']: v['name']
            for k, v in self.coco.cats.items()
        }
        # Fill the missing indexes
        na_cnt = 0
        for i in range(1, max(self.idx_to_class.keys())):  # 0 is reserved for background
            if i not in self.idx_to_class.keys():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Class to index dictionary
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        # Reduce the labels
        self.reduce_labels = reduce_labels
        if self.reduce_labels:
            self.idx_to_class = {k-1: v for k, v in self.idx_to_class.items()}
            self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        # Set the output format
        self.out_fmt = out_fmt
        # Set Transformers processor
        if out_fmt == "transformers":
            if processor is None:
                raise ValueError("`processor` must be provided if `out_fmt` is 'transformers'")
            else:
                self.processor = processor
        else:
            self.processor = None
        # Check whether all the image sizes are the same
        image_transform = self.transforms if self.transforms is not None else self.transform
        self.same_img_size = validate_same_img_size(image_transform, self.processor)

    def _load_target(self, id: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        if self.reduce_labels:
            labels = [label-1 for label in labels]
        # Get the bounding boxes
        boxes = [obj['bbox'] for obj in target_src]
        return boxes, labels
    
    def _convert_target(self, boxes, labels, id):
        """Convert COCO to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(size=(0,), dtype=torch.int64)
        # Convert the bounding boxes
        boxes = [[int(k) for k in _convert_bbox_xywh_to_xyxy(*box)]
                  for box in boxes]
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Get the image path
        image_path = self.coco.loadImgs(id)[0]["file_name"]
        target = {'boxes': boxes, 'labels': labels, 'image_path': os.path.join(self.root, image_path)}
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        if not isinstance(index, int):
            raise ValueError(f"Index must be of type integer, got {type(index)} instead.")

        id = self.ids[index]
        image = self._load_image(id)
        boxes, labels = self._load_target(id)

        # Apply transforms
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

        # Output as TorchVision format
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format
        # If the output image sizes are different, `processor.pad(pixel_values, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, index, self.processor, out_fmt='detr')
