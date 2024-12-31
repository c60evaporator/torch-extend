from typing import Tuple, Any
from torchvision.datasets import CIFAR10
import albumentations as A
from PIL import Image
import numpy as np

class CIFAR10TV(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                img = self.transform(image=np.array(img))["image"]  # For Albumentations
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
