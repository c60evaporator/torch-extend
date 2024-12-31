from typing import TypedDict
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from abc import abstractmethod

from ..base import TorchVisionDataModule
from torch_extend.display.detection import show_bounding_boxes

###### Annotation Validation TypeDicts ######
class ImageValidationResult(TypedDict):
    img_id: int
    img_path: str
    img_width: int
    img_height: int
    n_boxes: int
    anomaly: bool
    anomaly_box_width: bool
    anomaly_box_height: bool

class BoxValidationResult(TypedDict):
    img_id: int
    img_path: str
    label: int
    label_name: str
    bbox: list[float]
    box_width: float
    box_height: float
    anomaly: bool
    anomaly_box_width: bool
    anomaly_box_height: bool

###### Main Class ######
class DetectionDataModule(TorchVisionDataModule):
    def __init__(self, batch_size, num_workers,
                 dataset_name,
                 transforms=None, transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name, transforms, transform, target_transform)
        self.class_to_idx = None
        self.idx_to_class = None

    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    ###### Dataset Methods ######
    @abstractmethod
    def _get_datasets(self, ignore_transforms):
        """Dataset initialization"""
        raise NotImplementedError
    
    def _setup(self):
        self.train_dataset, self.val_dataset, self.test_dataset = self._get_datasets()
        if 'class_to_idx' in vars(self.train_dataset):
            self.class_to_idx = self.train_dataset.class_to_idx
        if 'idx_to_class' in vars(self.train_dataset):
            self.idx_to_class = self.train_dataset.idx_to_class
    
    def train_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def val_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self) -> list[str]:
        """Create dataloader"""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
    
    ###### Display methods ######
    def _show_image_and_target(self, img, target, denormalize=True, ax=None, anomaly_indices=None):
        """Show the image and the target"""
        if denormalize:  # Denormalize if normalization is included in transforms
            img = self._denormalize_image(img)
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        boxes, labels = target['boxes'], target['labels']
        show_bounding_boxes(img, boxes, labels=labels,
                            idx_to_class=self.idx_to_class, 
                            anomaly_indices=anomaly_indices, ax=ax)

    ###### Validation methods ######
    @abstractmethod
    def _output_filtered_annotation(self, del_img_ids, output_dir, image_set):
        """Output an annotation file whose anomaly images are excluded"""
        raise NotImplementedError

    def _validate_annotation(self, imgs, targets, anomaly_save_path, use_instance_loader):
        """Validate the annotations"""
        img_validations = []
        box_validations = []
        for img, target in zip(imgs, targets):
            # Image information
            img_result: ImageValidationResult = {}
            image_id = int(os.path.splitext(os.path.basename(target['image_path']))[0])
            img_result['image_id'] = image_id
            img_result['image_path'] = target['image_path']
            img_result['image_width'] = img.size()[-1]
            img_result['image_height'] = img.size()[-2]
            img_result['n_boxes'] = len(target['boxes'])
            img_result['anomaly'] = False
            anomaly_indices = []
            # Bounding box validation
            for i_box, (box, label) in enumerate(zip(target['boxes'], target['labels'])):
                box_list = box.tolist()
                box_result: BoxValidationResult = {}
                box_result['image_id'] = image_id
                box_result['image_path'] = target['image_path']
                box_result['label'] = label.item()
                box_result['label_name'] = str(box_result['label']) if self.idx_to_class is None else self.idx_to_class[box_result['label']]
                box_result['bbox'] = box_list
                box_result['box_width'] = box_list[2] - box_list[0]
                box_result['box_height'] = box_list[3] - box_list[1]
                # Negative box width
                box_result['anomaly_box_width'] = box_result['box_width'] <= 0
                # Negative box width
                box_result['anomaly_box_height'] = box_result['box_height'] <= 0
                # Final anomaly judgement
                box_result['anomaly'] = box_result['anomaly_box_width'] or box_result['anomaly_box_height']
                if box_result['anomaly']:
                    img_result['anomaly'] = True
                    anomaly_indices.append(i_box)
                box_validations.append(box_result)
            # Save the anomaly image
            if img_result['anomaly']:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                self._show_image_and_target(img, target, denormalize=use_instance_loader, ax=ax, anomaly_indices=anomaly_indices)
                fig.savefig(f'{anomaly_save_path}/{os.path.basename(target["image_path"])}')
            img_validations.append(img_result)
        return img_validations, box_validations
    
    def validate_annotation(self, result_path=None, use_instance_loader=False):
        """Validate the annotations"""
        if self.train_dataset is None or self.val_dataset is None:
            raise RuntimeError('Run the `setup()` method before the validation')
        if result_path == None:
            result_path = f'./ann_validation/{self.dataset_name}'
        # Get the datasets for the annotation validation
        if use_instance_loader:
            trainloader = self.train_dataloader()
            valloader = self.val_dataloader()
        else:
            trainset, valset, _ = self._get_datasets(ignore_transforms=True)
            trainloader = DataLoader(trainset, batch_size=1, 
                                    shuffle=False, num_workers=self.num_workers,
                                    collate_fn=self.collate_fn)
            valloader = DataLoader(valset, batch_size=1, 
                                shuffle=False, num_workers=self.num_workers,
                                collate_fn=self.collate_fn)
        os.makedirs(result_path, exist_ok=True)
        train_anomaly_path = f'{result_path}/anomaly_images/train'
        os.makedirs(train_anomaly_path, exist_ok=True)
        val_anomaly_path = f'{result_path}/anomaly_images/val'
        os.makedirs(val_anomaly_path, exist_ok=True)

        # Validate the annotation of train_dataset
        print('Validate the annotations of train_dataset')
        start = time.time()
        train_img_result: list[ImageValidationResult] = []
        train_box_result: list[BoxValidationResult] = []
        for i, (imgs, targets) in enumerate(trainloader):
            img_validations, box_validations = self._validate_annotation(imgs, targets, train_anomaly_path, use_instance_loader)
            train_img_result.extend(img_validations)
            train_box_result.extend(box_validations)
            if i%100 == 0:  # Show progress every 100 times
                print(f'Validating the annotations of train_dataset: {i}/{len(trainloader)}, elapsed_time: {time.time() - start}')
        # Output the validation result
        df_train_img_result = pd.DataFrame(train_img_result)
        df_train_img_result.to_csv(f'{result_path}/train_img_validation.csv')
        df_train_box_result = pd.DataFrame(train_box_result)
        df_train_box_result.to_csv(f'{result_path}/train_box_validation.csv')
        plt.show()
        # Output a new annotation file whose anomaly images are excluded
        train_anom_img_ids = df_train_img_result[df_train_img_result['anomaly']]['image_id'].tolist()
        self._output_filtered_annotation(train_anom_img_ids, result_path, 'train')
            
        # Validate the annotation of val_dataset
        print('Validate the annotations of val_dataset')
        start = time.time()
        val_img_result: list[ImageValidationResult] = []
        val_box_result: list[BoxValidationResult] = []
        for i, (imgs, targets) in enumerate(valloader):
            img_validations, box_validations = self._validate_annotation(imgs, targets, val_anomaly_path, use_instance_loader)
            val_img_result.extend(img_validations)
            val_box_result.extend(box_validations)
            if i%100 == 0:  # Show progress every 100 times
                print(f'Validating the annotations of val_dataset: {i}/{len(valloader)}, elapsed_time: {time.time() - start}')
        # Output the validation result
        df_val_img_result = pd.DataFrame(val_img_result)
        df_val_img_result.to_csv(f'{result_path}/val_img_validation.csv')
        df_val_box_result = pd.DataFrame(val_box_result)
        df_val_box_result.to_csv(f'{result_path}/val_box_validation.csv')
        plt.show()
        # Output a new annotation file whose anomaly images are excluded
        val_anom_img_ids = df_val_img_result[df_val_img_result['anomaly']]['image_id'].tolist()
        self._output_filtered_annotation(val_anom_img_ids, result_path, 'val')
