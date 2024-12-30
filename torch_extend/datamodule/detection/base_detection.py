from torch.utils.data import DataLoader

from ..base import TorchVisionDataModule

class DetectionDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 transforms=None, transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, transforms, transform, target_transform)

    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def train_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
