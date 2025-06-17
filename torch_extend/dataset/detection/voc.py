from typing import Any, Callable, List, Dict, Optional, Tuple, Literal
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.voc import DATASET_YEAR_DICT
from transformers import BaseImageProcessor
import albumentations as A
import numpy as np
from PIL import Image
import collections
import os
import shutil

from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse

from .utils import DetectionOutput
from ...validate.common import validate_same_img_size
from ...data_converter.detection import convert_image_target_to_transformers

def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    """
    From torchvision.datasets.VOCDetection.parse_voc_xml
    """
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

class VOCBaseTV(VisionDataset):
    IDX_TO_CLASS = {
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }
    
    def _get_images_targets(self, root, image_set, splits_dir_name, target_dir_name, target_file_ext, aux_target_dir_name=None):
        # Get the path list of images and targets
        splits_dir = os.path.join(root, "ImageSets", splits_dir_name)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]
        # Get the images
        image_dir = os.path.join(root, "JPEGImages")
        images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # Get the targets
        target_dir = os.path.join(root, target_dir_name)
        targets = [os.path.join(target_dir, x + target_file_ext) for x in file_names]
        # Get the aux targets (Bounding boxes for instance segmentation)
        aux_targets = None
        if aux_target_dir_name is not None:
            aux_target_dir = os.path.join(root, aux_target_dir_name)
            aux_targets = [os.path.join(aux_target_dir, x + '.xml') for x in file_names]
            assert len(images) == len(aux_targets)

        assert len(images) == len(targets)
        return images, targets, aux_targets

    def __init__(
        self,
        root: str,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.image_set = image_set

        if download:
            # Download VOC2012 dataset
            url = DATASET_YEAR_DICT["2012"]["url"]
            filename = DATASET_YEAR_DICT["2012"]["filename"]
            md5 = DATASET_YEAR_DICT["2012"]["md5"]
            if not os.path.isfile(f'{root}/{filename}'):
                download_and_extract_archive(url, root, filename=filename, md5=md5)
                # Move the extracted files to the root directory
                base_dir = DATASET_YEAR_DICT["2012"]["base_dir"]
                voc_root = os.path.join(root, base_dir)
                if os.path.isdir(voc_root):
                    shutil.move(f'{voc_root}/Annotations', root)
                    shutil.move(f'{voc_root}/ImageSets', root)
                    shutil.move(f'{voc_root}/JPEGImages', root)
                    shutil.move(f'{voc_root}/SegmentationClass', root)
                    shutil.move(f'{voc_root}/SegmentationObject', root)
                    # Delete the parent of voc_root
                    shutil.rmtree(os.path.join(root, os.path.dirname(base_dir)))

        if not os.path.isdir(root):
            raise RuntimeError("Dataset not found or corrupted")
        
        # Get the images and bboxes for Object Detection
        self.images_detection, self.bboxes_detection, _ = self._get_images_targets(root, image_set, "Main", "Annotations", ".xml")

        # Get the images and masks for Semantic Segmentation
        if "SegmentationClass" in os.listdir(root):
            self.images_semantic, self.masks_semantic, _ = self._get_images_targets(root, image_set, "Segmentation", "SegmentationClass", ".png")

        # Get the images, masks, and bboxes for Instance Segmentation
        if "SegmentationObject" in os.listdir(root):
            self.images_instance, self.masks_instance, self.bboxes_instance = self._get_images_targets(
                root, image_set, "Segmentation", "SegmentationObject", ".png", aux_target_dir_name="Annotations")

class VOCDetection(VOCBaseTV, DetectionOutput):
    """
    Dataset from Pascal VOC format to Torchvision or Transformers format with image path

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
        If True, the label 0 is regarded as the background and all the labels will be reduced by 1. Also, the class_to_idx will be updated accordingly.

        For example, if the labels are [1, 3] and the class_to_idx is {1: 'aeroplane', 2: 'bicycle', 3: 'bird'}, the labels will be [0, 2] and the class_to_idx will be {0: 'aeroplane', 1: 'bicycle', 2: 'bird'}.
    processor : callable, optional
        An image processor instance for HuggingFace Transformers. Only available if ``out_fmt="transformers"``.

    The dataset folder structure should be like this:
    ```
    root/  <-- This folder should be set as `root`
        ├── Annotations/
        │   ├── image001.xml
        │   ├── image002.xml
        │   └── ...
        ├── ImageSets/
        │   ├── Main/
        │   │   ├── train.txt
        │   │   ├── val.txt
        │   │   └── ...
        │   └── (Segmentation/)
        │       └── ...
        ├── JPEGImages/
        │   ├── image001.jpg
        │   ├── image002.jpg
        │   └── ...
        ├── (SegmentationClass/)
        │   └── ...
        └── (SegmentationObject/)
            └── ...
    ```
    """

    def __init__(
        self, 
        root: str,
        idx_to_class : Dict[int, str] = None,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        image_set: Literal["train", "val"] = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        reduce_labels: bool = False,
        processor: Optional[BaseImageProcessor] = None
    ):
        super().__init__(root, image_set, download, transform, target_transform, transforms)
        self.ids = os.listdir(root)
        # Index to class dictionary
        if idx_to_class is None:
            self.idx_to_class = self.IDX_TO_CLASS
        else:
            self.idx_to_class = idx_to_class
        # Class to index dictionary
        self.class_to_idx_orig = {v: k for k, v in self.idx_to_class.items()}
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

    def __len__(self) -> int:
        return len(self.images_detection)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_detection[index]).convert("RGB")
    
    def _load_voc_bboxes(self, bbox_xml: str) -> Dict[str, Any]:
        # Read bounding box XML file 
        target = parse_voc_xml(ET_parse(bbox_xml).getroot())
        return target['annotation']['object']
    
    def _load_target(self, index: int) -> List[Any]:
        # Read XML file 
        objects = self._load_voc_bboxes(self.bboxes_detection[index])
        # Get the labels
        labels = [self.class_to_idx_orig[obj['name']] for obj in objects]
        # Get the bounding boxes
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        return boxes, labels
    
    def _convert_target(self, boxes, labels, index):
        """Convert VOC to TorchVision format"""
        # Get the labels
        labels = torch.tensor(labels, dtype=torch.int64) if len(labels) > 0 else torch.zeros(size=0, dtype=torch.int64)
        # Convert the bounding boxes
        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros(size=(0, 4), dtype=torch.float32)
        # Output as Torchvision format
        target = {'boxes': boxes,
                  'labels': labels,
                  'image_path': self.images_detection[index]}
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        boxes, labels = self._load_target(index)

        # Apply transforms
        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), bboxes=boxes, class_labels=labels)
                image = transformed['image']
                target = self._convert_target(transformed['bboxes'], 
                                              transformed['class_labels'], index)
            # TorchVision transforms
            else:
                converted_target = self._convert_target(boxes, labels, index)
                image, target = self.transforms(image, converted_target)
        # No transformation
        else:
            target = self._convert_target(boxes, labels, index)

        # Output as TorchVision format
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format
        # If the output image sizes are different, `processor.pad(pixel_values, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, index, self.processor, out_fmt='detr')
