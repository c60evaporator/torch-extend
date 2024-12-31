from lightning.pytorch import LightningDataModule
import albumentations as A
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class TorchVisionDataModule(LightningDataModule, ABC):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 transforms=None, transform=None, target_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        # Create transform
        if transforms is None and transform is None and target_transform is None:
            self.transform = None
            self.target_transform = None
            self.transforms = self.default_transforms
        else:
            self.transform = transform
            self.target_transform = target_transform
            self.transforms = transforms
        # Check whether all the image sizes are the same
        self.same_img_size = False
        image_transform = self.transforms if self.transforms is not None else self.transform
        for tr in image_transform:
            if (isinstance(tr, v2.Resize) and len(tr.size) == 2) or isinstance(tr, A.Resize):
                self.same_img_size = True
                break
        # Other
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    ###### Dataset Methods ######
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
    
    def test_dataloader(self) -> list[str]:
        """Create test dataloader"""
        raise NotImplementedError
    
    ###### Display methods ######
    def _denormalize_image(self, img):
        """Denormalize the image for showing it"""
        image_transform = self.transforms if self.transforms is not None else self.transform
        for tr in image_transform:
            if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
                reverse_transform = v2.Compose([
                    v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                        std=[1/std for std in tr.std])
                ])
                return reverse_transform(img)
        return img
    
    @abstractmethod
    def _show_image_and_target(self, img, target, denormalize=True, ax=None):
        """Show the image and the target"""
        raise NotImplementedError

    def show_first_minibatch(self, image_set='train'):
        # Check whether all the image sizes are the same
        if image_set == 'train':
            loader = self.train_dataloader()
        elif image_set == 'val':
            loader = self.val_dataloader()
        else:
            raise RuntimeError('The `image_set` argument should be "train" or "val"')
        train_iter = iter(loader)
        imgs, targets = next(train_iter)

        for i, (img, target) in enumerate(zip(imgs, targets)):
            self._show_image_and_target(img, target)
            plt.show()
    
    ###### Validation Methods ######
    
    ###### Transform Methods ######
    @property
    @abstractmethod
    def default_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        raise NotImplementedError
