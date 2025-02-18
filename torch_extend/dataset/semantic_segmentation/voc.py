from typing import Any, Callable, List, Dict, Optional, Tuple
import albumentations as A
import numpy as np
from PIL import Image
import os

from ...dataset.detection.voc import VOCBaseTV
from .utils import SemanticSegOutput

class VOCSemanticSegmentation(VOCBaseTV, SemanticSegOutput):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Semantic Segmentation Dataset.

    Parameters
    ----------
    root : str
        Root directory of the VOC Dataset.
    idx_to_class : Dict[int, str]
        A dict which indicates the conversion from the label indices to the label names
    border_idx : int
        The index of the border in the target mask.
    bg_idx : int
        The index of the background in the target mask.
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
    """

    def __init__(
        self,
        root: str,
        idx_to_class: Dict[int, str] = None,
        border_idx: int = 255,
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
        self.border_idx = border_idx
        self.bg_idx = bg_idx

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
                image, target = self.transforms(image, target)

        # Postprocessing of the target
        target = target.squeeze(0).long()  # Convert to int64

        return image, target
    
    def get_image_target_path(self, index: int):
        """Get the image and target path of the dataset."""
        return self.images_semantic[index], self.masks_semantic[index]
