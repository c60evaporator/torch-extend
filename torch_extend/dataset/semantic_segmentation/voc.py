from typing import Any, Callable, List, Dict, Optional, Tuple, Literal
import torch
import albumentations as A
import numpy as np
from PIL import Image
import os
from transformers import BaseImageProcessor

from ...dataset.detection.voc import VOCBaseTV
from .utils import SemanticSegOutput
from ...validate.common import validate_same_img_size
from ...data_converter.semantic_segmentation import convert_image_target_to_transformers

class VOCSemanticSegmentation(VOCBaseTV, SemanticSegOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Semantic Segmentation Dataset.

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
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    bg_idx : int
        The index of the background in the target mask.
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
        processor: Optional[BaseImageProcessor] = None,
        bg_idx: int = 0,
        border_idx: int = 255,
    ):
        super().__init__(root, image_set, download, transform, target_transform, transforms)
        self.ids = os.listdir(root)
        # Index to class dictionary
        if idx_to_class is None:
            self.idx_to_class = self.IDX_TO_CLASS
        else:
            self.idx_to_class = idx_to_class
        # Class to index dictionary
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        # Set the border and background index
        self.orig_border_idx = border_idx
        self.border_idx = border_idx
        self.orig_bg_idx = bg_idx
        self.bg_idx = bg_idx
        # Set the output format
        self.out_fmt = out_fmt
        # Set Transformers processor
        if out_fmt == "transformers":
            if processor is None:
                raise ValueError("`processor` must be provided if `out_fmt` is 'transformers'")
            else:
                self.processor = processor
                # If `processor.do_reduce_labels=True`, `idx_to_class` is also reduced and border index is set to 255
                if self.processor.do_reduce_labels:
                    self.idx_to_class = {k-1: v for k, v in self.idx_to_class.items()}
                    self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
                    self.bg_idx = None  # Background index is ignored if `do_reduce_labels=True`
                    self.border_idx = 255  # Background is regared as border (255 in SegFormer) if `do_reduce_labels=True`
        else:
            self.processor = None
        # Chech whether all the image sizes are the same
        image_transform = self.transforms if self.transforms is not None else self.transform
        self.same_img_size = validate_same_img_size(image_transform, self.processor)

    def __len__(self) -> int:
        return len(self.images_semantic)
    
    def _load_image(self, index: int) -> Image.Image:
        return Image.open(self.images_semantic[index]).convert("RGB")
    
    def _load_target(self, index: int) -> List[Any]:
        return Image.open(self.masks_semantic[index])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        image = self._load_image(index)
        mask = self._load_target(index)

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), mask=np.asarray(mask).copy())
                image, target = transformed['image'], transformed['mask']
            # TorchVision transforms
            else:
                image, target = self.transforms(image, mask)
        
        # Convert to Torch Tensor if not
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        if not isinstance(target, torch.Tensor):
            if not isinstance(target, np.ndarray):
                target = np.array(target)
            target = torch.tensor(target)

        # Postprocessing of the target
        target = target.squeeze(0).long()  # Convert to int64

        # Output as TorchVision format
        if self.out_fmt == "torchvision":
            return image, target

        # Output as Transformers format (Original border mask is replaced with background)
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, self.processor, self.same_img_size,
                                                        orig_bg_idx=self.orig_bg_idx, orig_border_idx=self.orig_border_idx, out_fmt="segformer")
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_semantic[index], self.masks_semantic[index]
