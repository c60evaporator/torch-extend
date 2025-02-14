from typing import Any, Callable, List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from ...dataset.detection.voc import VOCBaseTV
from .utils import SemanticOutput

class VOCSemanticTV(VOCBaseTV, SemanticOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a transformed version. E.g, ``transforms.PILToTensor``
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : callable, optional
        A function/transform that takes input sample and its target as entry and returns a transformed version.
    albumentations_transform : albumentations.Compose, optional
        An Albumentations function that is applied to both the PIL image and the target. This transform is applied in advance of `transform` and `target_transform`. (https://stackoverflow.com/questions/58215056/how-to-use-torchvision-transforms-for-data-augmentation-of-segmentation-task-in)
    """
    _SPLITS_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    def __init__(
        self,
        root: str,
        idx_to_class: Dict[int, str],
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        albumentations_transform: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, transform, target_transform, transforms)

        # Additional
        self.idx_to_class = idx_to_class
        self.albumentations_transform = albumentations_transform

    def __len__(self) -> int:
        return len(self.images_semantic)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        image = Image.open(self.images_semantic[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.albumentations_transform is not None:
            # https://albumentations.ai/docs/examples/pytorch_semantic_segmentation/
            A_transformed = self.albumentations_transform(image=np.array(image), mask=np.asarray(target).copy())
            image = A_transformed['image']
            target = A_transformed['mask']

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Postprocessing of the target
        target = target.squeeze(0).long()  # Convert to int64
        target[target == 255] = len(self.idx_to_class)  # Replace the border of the target mask

        return image, target
