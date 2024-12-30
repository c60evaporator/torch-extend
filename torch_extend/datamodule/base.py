from lightning.pytorch import LightningDataModule
import albumentations as A
from torchvision.transforms import v2
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class TorchVisionDataModule(LightningDataModule, ABC):
    def __init__(self, batch_size, num_workers,
                 transforms=None, transform=None, target_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Create transform
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = self.default_transforms
        # Check whether all the image sizes are the same
        self.same_img_size = False
        for tr in self.transforms.transforms:
            if (isinstance(tr, v2.Resize) and len(tr.size) == 2) or isinstance(tr, A.Resize):
                self.same_img_size = True
                break
        # Other
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @abstractmethod
    def _setup(self):
        """Dataset initialization"""
        raise NotImplementedError
    
    def setup(self, stage=None):
        self._setup()
    
    def train_dataloader(self) -> list[str]:
        """Create train dataloader"""
        raise NotImplementedError
    
    def val_dataloader(self) -> list[str]:
        """Create validation dataloader"""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_transforms(self) -> list[v2.Compose | A.Compose]:
        """Default transforms for preprocessing"""
        raise NotImplementedError
