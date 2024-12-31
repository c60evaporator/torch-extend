import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
import cv2

from .base_classification import ClassificationDataModule
from ...dataset.classification.cifar import CIFAR10TV

###### Main Class ######
class CIFAR10DataModule(ClassificationDataModule):
    def __init__(self, batch_size, num_workers,
                 root, download=True,
                 dataset_name='CIFAR10',
                 transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name, transform, target_transform)
        self.root = root
        self.download = download
    
    ###### Dataset Methods ######
    def _get_datasets(self):
        """Dataset initialization"""
        train_dataset = CIFAR10TV(
            self.root,
            train=True,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download
        )
        val_dataset = CIFAR10TV(
            self.root,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.download
        )
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
        return train_dataset, val_dataset, None
    
    ###### Validation methods ######
    def validate_annotation(self, result_path='./ann_validation', use_instance_loader=False):
        """Validate the annotations"""
        pass

    ###### Transform Methods ######
    @property
    def default_transform(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)"""
        return A.Compose([
            A.Resize(32,32),
            A.HorizontalFlip(),
            A.Rotate(limit=10, interpolation=cv2.INTER_NEAREST),
            A.Affine(rotate=0, shear=10, scale=(0.8,1.2)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
        ])