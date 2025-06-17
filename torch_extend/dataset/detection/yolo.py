from typing import Any, Callable, List, Dict, Optional, Tuple, Literal
import torch
from torchvision.datasets import VisionDataset
from transformers import BaseImageProcessor
import albumentations as A
import numpy as np
from PIL import Image
import os
import yaml

from .utils import DetectionOutput
from ...validate.common import validate_same_img_size
from ...data_converter.detection import _convert_bbox_centerxywh_to_xyxy, convert_image_target_to_transformers

class YOLODetection(VisionDataset, DetectionOutput):
    """
    Dataset from YOLO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to dataset root folder
    idx_to_class : Dict[int, str], optional
        A dict which indicates the conversion from the label indices to the label names. Should be None if `data_yaml` is provided.
    data_yaml : str, optional
        Path to the data YAML file which contains the dataset information. Should be None if `idx_to_class` is provided.
    out_fmt : Literal["torchvision", "transformers"]
        The output format of the image and target, either ``"torchvision"`` or ``"transformers"``.
    image_set : str
        Select the image_set to use, ``"train"`` or ``"val"``.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``.
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
    root/  <-- This folder should be set as `root`
        ├── images/
        │   ├── train/
        │   │   ├── image001.jpg
        │   │   ├── image002.jpg
        │   │   └── ...
        │   └── val/
        │       ├── image001.jpg
        │       ├── image002.jpg
        │       └── ...
        └── annotations/
            ├── train/
            │   ├── image001.txt
            │   ├── image002.txt
            │   └── ...
            └── val/
                ├── image001.txt
                ├── image002.txt
                └── ...
    ```

    The annotation text files should be in the YOLO format, where each line contains:
    ```
    <class_id> <x_center> <y_center> <width> <height>
    ```

    The format of `data_yaml` should be like this:
    ```yaml
    train: images/train
    val: images/val
    nc: 3  # number of classes
    names:  # class names
      0: 'person'
      1: 'bicycle'
      2: 'car'
    ```
    """
    def __init__(
        self, 
        root: str,
        idx_to_class: Dict[int, str] = None,
        data_yaml: str = None,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        image_set: Literal["train", "val"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        reduce_labels: bool = False,
        processor: Optional[BaseImageProcessor] = None
    ):
        if data_yaml is None and idx_to_class is None:
            raise ValueError("Either `data_yaml` or `idx_to_class` must be provided.")
        if data_yaml is not None and idx_to_class is not None:
            raise ValueError("Only one of `data_yaml` or `idx_to_class` can be provided.")
        
        super().__init__(root, transforms, transform, target_transform)
        self.image_set = image_set

        # Get the images and target paths
        self.images, self.targets = self._get_images_targets(root, image_set)
        self.ids = [os.path.basename(p) for p in self.images]

        # Index to class dictionary
        if idx_to_class is None:
            # Load the data YAML file
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
            self.idx_to_class = {i: name for i, name in enumerate(data['names'])}
        else:
            self.idx_to_class = idx_to_class
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

    def _get_images_targets(self, root, image_set):
        """ Get the images and targets from the dataset root directory."""
        # Get the images
        image_dir = os.path.join(root, 'images', image_set)
        images = [os.path.join(image_dir, x) 
                  for x in os.listdir(image_dir) 
                  if x.endswith(('.jpg', '.png', '.jpeg'))]
        # Estimated annotations paths
        ann_dir = f'{root}/annotations/{image_set}'
        estimated_anns = [f'{ann_dir}/{os.path.splitext(os.path.basename(p))[0]}.txt'
                          for p in images]
        # Check if the annotations exist
        filtered_images = []
        anns = []
        for image, ann in zip(images, estimated_anns):
            if not os.path.exists(os.path.join(ann_dir, ann)):
                print(f"Warning: Annotation file {ann} does not exist in {ann_dir}. Skipping this image.")
            else:
                filtered_images.append(image)
                anns.append(ann)
        return filtered_images, anns
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images[index]).convert("RGB")
    
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
        return boxes, labels
    
    def _convert_target(self, boxes, labels, index: int, w: int, h: int):
        """Convert COCO to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(size=0, dtype=torch.int64)
        # Convert the bounding boxes from [x_c, y_c, w, h] to [xmin, ymin, xmax, ymax]
        boxes = [[int(k) for k in _convert_bbox_centerxywh_to_xyxy(*box)]
                  for box in boxes]
        # Convert normalized coordinates to raw coordinates
        boxes = torch.tensor(boxes) * torch.tensor([w, h, w, h], dtype=torch.float32) if len(boxes) > 0 \
                else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Get the image path
        target = {'boxes': boxes,
                  'labels': labels,
                  'image_path': self.images[index]}
        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        boxes, labels = self._load_target(id)
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

        # Output as TorchVision format
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format
        # If the output image sizes are different, `processor.pad(pixel_values, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, index, self.processor, out_fmt='detr')
