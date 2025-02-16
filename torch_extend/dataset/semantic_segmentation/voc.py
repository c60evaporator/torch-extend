from typing import Any, Callable, List, Dict, Optional, Tuple
import albumentations as A
import numpy as np
from PIL import Image
import os

from ...dataset.detection.voc import VOCBaseTV
from .utils import SemanticSegOutput

class VOCSemanticSegmentation(VOCBaseTV, SemanticSegOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    image_set : str
        Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``.
    border_idx : int
        The index of the border in the target mask.
    bg_idx : int
        The index of the background in the target mask.
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
        idx_to_class: Dict[int, str] = None,
        border_idx: int = None,
        bg_idx: int = 0,
        image_set: str = "train",
        download: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, image_set, download, transform, target_transform, transforms)
        if idx_to_class is None:
            self.idx_to_class = self.IDX_TO_CLASS
        else:
            self.idx_to_class = idx_to_class
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.ids = os.listdir(root)
        self.border_idx = border_idx if border_idx is not None else len(self.idx_to_class)
        self.bg_idx = bg_idx

    def __len__(self) -> int:
        return len(self.images_semantic)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """"""
        image = Image.open(self.images_semantic[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            # Albumentation transforms
            if isinstance(self.transforms, A.Compose):
                transformed = self.transforms(image=np.array(image), mask=np.asarray(target).copy())
                image, target = transformed['image'], transformed['mask']
            # TorchVision transforms
            else:
                image, target = self.transforms(image, target)

        # Postprocessing of the target
        target = target.squeeze(0).long()  # Convert to int64
        target[target == 255] = self.border_idx  # Replace the border index of the target mask

        return image, target
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_semantic[index], self.masks[index]
