from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets import CocoDetection
import numpy as np
from PIL import Image

from ...data_converter.semantic_segmentation import convert_polygon_to_mask, merge_masks
from .utils import SemanticOutput

class CocoSemanticTV(CocoDetection, SemanticOutput):
    """
    Dataset from COCO format to Torchvision format with image path

    Parameters
    ----------
    root : str
        Path to images folder
    annFile : str
        Path to annotation text file folder
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    albumentations_transform : albumentations.Compose, optional
        An Albumentations function that is applied to both the PIL image and the target. This transform is applied in advance of `transform` and `target_transform`. (https://stackoverflow.com/questions/58215056/how-to-use-torchvision-transforms-for-data-augmentation-of-segmentation-task-in)
    """
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        albumentations_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, target_transform, transforms)
        self.idx_to_class = {
            v['id']: v['name']
            for k, v in self.coco.cats.items()
        }
        self.albumentations_transform = albumentations_transform

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
        target = self._load_target(id, image.size[1], image.size[0])
        target = Image.fromarray(target)

        if self.albumentations_transform is not None:
            A_transformed = self.albumentations_transform(image=np.array(image), mask=np.asarray(target).copy())
            image = A_transformed['image']
            target = A_transformed['mask']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Postprocessing of the target
        target = target.squeeze(0).long()  # Convert to int64

        return image, target
