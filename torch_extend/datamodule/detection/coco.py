import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2
import json

from .base_detection import DetectionDataModule
from ...dataset.detection.coco import CocoDetectionTV

class CocoDataModule(DetectionDataModule):
    def __init__(self, batch_size, num_workers, 
                 root, train_dir='train2017', val_dir='val2017',
                 train_annFile=None, val_annFile=None,
                 dataset_name='COCO',
                 transforms=None, transform=None, target_transform=None):
        super().__init__(batch_size, num_workers, dataset_name, transforms, transform, target_transform)
        self.root = root
        self.train_dir = train_dir
        self.val_dir = val_dir
        # Annotation files
        if train_annFile is not None:
            self.train_annFile = train_annFile
        else:
            self.train_annFile = f'{self.root}/annotations/instances_{self.train_dir}.json'
        if val_annFile is not None:
            self.val_annFile = val_annFile
        else:
            self.val_annFile = f'{self.root}/annotations/instances_{self.val_dir}.json'

    ###### Dataset Methods ######
    def _get_datasets(self, ignore_transforms=False):
        """Dataset initialization"""
        train_dataset = CocoDetectionTV(
            f'{self.root}/{self.train_dir}',
            annFile=self.train_annFile,
            transforms=self.transforms if not ignore_transforms else None,
            transform=v2.ToTensor() if ignore_transforms else None
        )
        val_dataset = CocoDetectionTV(
            f'{self.root}/{self.val_dir}',
            annFile=self.val_annFile, 
            transforms=self.transforms if not ignore_transforms else None,
            transform=v2.ToTensor() if ignore_transforms else None
        )
        return train_dataset, val_dataset, None
    
    ###### Validation Methods ######
    def _output_filtered_annotation(self, del_img_ids, output_dir, image_set='train'):
        if image_set=='train':
            coco_dataset = self.train_dataset.coco.dataset
        elif image_set == 'val':
            coco_dataset = self.val_dataset.coco.dataset
        else:
            raise RuntimeError('The `image_set` argument should be "train" or "val"')
        # Load the coco fields
        coco_info = coco_dataset['info']
        coco_licenses = coco_dataset['licenses']
        coco_images = coco_dataset['images']
        coco_annotations = coco_dataset['annotations']
        coco_categories = coco_dataset['categories']
        # Filter the images
        filtered_images = [image for image in coco_images if image['id'] not in del_img_ids]
        # Filter the annotations
        filtered_annotations = [ann for ann in coco_annotations if ann['image_id'] not in del_img_ids]
        # Output the filtered annotation JSON file
        filtered_coco = {
            'info': coco_info,
            'licenses': coco_licenses,
            'images': filtered_images,
            'annotations': filtered_annotations,
            'categories': coco_categories
        }
        with open(f'{output_dir}/instances_{image_set}_filtered.json', 'w') as f:
            json.dump(filtered_coco, f, indent=None)
        
    ###### Transform Methods ######
    @property
    def default_transforms(self) -> v2.Compose | A.Compose:
        """Default transforms for preprocessing"""
        return A.Compose([
            A.Resize(640, 640),  # Resize the image to (640, 640)
            A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # Normalization (mean and std of the ImageNet dataset for normalizing)
            ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
        ], bbox_params=A.BboxParams(format='coco'))
