from typing import Any, Callable, List, Optional, Tuple, Literal
import torch
import torchvision.datasets as ds
import albumentations as A
from transformers import BaseImageProcessor
import numpy as np
from PIL import Image

from ...data_converter.semantic_segmentation import convert_polygon_to_mask, merge_masks
from ...data_converter.semantic_segmentation import convert_image_target_to_transformers
from ...validate.common import validate_same_img_size
from .utils import SemanticSegOutput

class CocoSemanticSegmentation(ds.CocoDetection, SemanticSegOutput):
    """
    Dataset from COCO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
    out_fmt : Literal["torchvision", "transformers"]
        The output format of the image and target, either ``"torchvision"`` or ``"transformers"``.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    bg_idx : int
        The index of the background in the target mask.
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        out_fmt: Literal["torchvision", "transformers"] = "torchvision",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        processor: Optional[BaseImageProcessor] = None,
        bg_idx: int = 0,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        # Index to class dictionary
        self.idx_to_class = {
            v['id']: v['name']
            for k, v in self.coco.cats.items()
        }
        na_cnt = 0
        for i in range(max(self.idx_to_class.keys())):
            if i not in self.idx_to_class.keys():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        # Class to index dictionary
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        # Set the background index
        self.bg_idx = bg_idx
        self.border_idx = None  # No border in COCO dataset
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
                    self.border_idx = 255  # Background is regared as border if `do_reduce_labels=True`
        else:
            self.processor = None
        # Chech whether all the image sizes are the same
        image_transform = self.transforms if self.transforms is not None else self.transform
        self.same_img_size = validate_same_img_size(image_transform, self.processor)

    def _load_target(self, id: int, height: int, width: int) -> List[Any]:
        target_src = self.coco.loadAnns(self.coco.getAnnIds(id))
        # Get the labels
        labels = [obj['category_id'] for obj in target_src]
        labels = np.array(labels)
        # Get the segmentation polygons
        segmentations = [obj['segmentation'] for obj in target_src]
        # Convert the polygons to masks
        masks = convert_polygon_to_mask(segmentations, height, width)
        # Merge the masks
        target = merge_masks(labels, masks)
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        w, h = image.size
        target = self._load_target(id, h, w)
        target = Image.fromarray(target)

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), mask=np.asarray(target).copy())
                image, target = transformed['image'], transformed['mask']
            # TorchVision transforms
            else:
                image, target = self.transforms(image, target)

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

        # Output as Transformers format
        elif self.out_fmt == "transformers":
            return convert_image_target_to_transformers(image, target, self.processor, self.same_img_size, out_fmt="segformer")
