import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

from .base_detection import DetectionDataModule
from ...dataset.detection.coco import CocoDetectionTV

class CocoDataModule(DetectionDataModule):
    def __init__(self, batch_size, num_workers,
                 root, train_dir='train2017', val_dir='val2017',
                 transforms=None, transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, transforms, transform, target_transform)
        self.root = root
        self.train_dir = train_dir
        self.val_dir = val_dir

    def _setup(self):
        """Dataset initialization"""
        self.train_dataset = CocoDetectionTV(
            f'{self.root}/{self.train_dir}',
            annFile=f'{self.root}/annotations/instances_{self.train_dir}.json', 
            transforms=self.transforms
        )
        self.val_dataset = CocoDetectionTV(
            f'{self.root}/{self.val_dir}',
            annFile=f'{self.root}/annotations/instances_{self.val_dir}.json', 
            transforms=self.transforms
        )

    def default_transforms(self) -> list[v2.Compose | A.Compose]:
        """Default transforms for preprocessing"""
        return A.Compose([
            A.Resize(640, 640),  # Resize the image to (640, 640)
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalization (mean and std of the imagenet dataset for normalizing)
            ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
        ], bbox_params=A.BboxParams(format='coco'))
