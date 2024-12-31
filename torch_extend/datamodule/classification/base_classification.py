from torch.utils.data import DataLoader
import albumentations as A
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from abc import abstractmethod

from ..base import TorchVisionDataModule

###### Main Class ######
class ClassificationDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name, None, transform, target_transform)
        self.transform = transform if transform is not None else self.default_transform
        self.target_transform = target_transform if target_transform is not None else self.default_target_transform
        self.class_to_idx = None
        self.idx_to_class = None
    
    ###### Dataset Methods ######
    @abstractmethod
    def _get_datasets(self, ignore_transforms):
        """Dataset initialization"""
        raise NotImplementedError
    
    def _setup(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
    
    def train_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers)
    
    ###### Display methods ######    
    def _show_image_and_target(self, img, target, denormalize=True, ax=None):
        # If ax is None, use matplotlib.pyplot.gca()
        if ax is None:
            ax=plt.gca()
        # Denormalize if normalization is included in transforms
        if denormalize:
            img = self._denormalize_image(img)
        img_permute = img.permute(1, 2, 0)
        ax.imshow(img_permute)  # Display the image
        ax.set_title(f'label: {self.idx_to_class[target.item()] if self.idx_to_class is not None else target.item()}')

    ###### Validation methods ######
    def validate_annotation(self, result_path='./ann_validation', use_instance_loader=False):
        """Validate the annotations"""
        pass

    ###### Transform Methods ######
    @property
    @abstractmethod
    def default_transform(self) -> v2.Compose | A.Compose:
        """Default transform for preprocessing"""
        raise NotImplementedError
    
    @property
    def default_target_transform(self):
        """Default target transform for preprocessing"""
        None

    @property
    def default_transforms(self):
        """Default transforms for preprocessing"""
        None
