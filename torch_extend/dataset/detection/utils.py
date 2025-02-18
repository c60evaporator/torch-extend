import yaml
import os
import shutil
import json

from ...data_converter.detection import target_transform_from_torchvision

class DetectionOutput():
    def _get_images_targets(self):
        sample_size = self.__len__()
        images_targets = [self.__getitem__(idx) for idx in range(sample_size)]
        images = [image for image, target in images_targets]
        targets = [target for image, target in images_targets]
        return images, targets

    def get_coco_annotation(self, info_dict=None, licenses_list=None):
        if self.idx_to_class is None:
            raise AttributeError('The "idx_to_class" attribute should not be None if the output format is COCO')
        # Get target and images as the TorchVision format
        images, targets = self._get_images_targets()
        # Convert the target to COCO format
        targets = [
            target_transform_from_torchvision(target, out_format='coco')
            for target in targets
        ]
        # Create "images" field in the annotation file
        ann_images = [
            {
                'license': 4,
                'file_name': os.path.basename(image_fp),
                'coco_url': '',
                'height': image.height,
                'width': image.width,
                'date_captured': '',
                'flickr_url': '',
                'id': i_img,
            }
            for i_img, (image, image_fp) in enumerate(zip(images, self.images))
        ]
        # Create "annotations" field in the annotation file (TODO: Add "segmentation" field by cv2.approxPolyDP)
        ann_annotations = []
        obj_id_cnt = 0
        for i_img, target in enumerate(targets):
            for obj in target:
                obj_ann = {
                    'segmentation': [],
                    'area': obj['area'] if 'area' in obj.keys() else 0,
                    'iscrowd': obj['iscrowd'] if 'iscrowd' in obj.keys() else 0,
                    'image_id': obj['image_id'] if 'image_id' in obj.keys() else i_img,
                    'bbox': obj['boxes'],
                    'category_id': obj['category_id'] if 'category_id' in obj.keys() else 0,
                    'id': obj['id'] if 'id' in obj.keys() else obj_id_cnt,
                }
                obj_id_cnt += 1
                ann_annotations.append(obj_ann)
        # Create "categories" field in the annotation file
        ann_categories = [
            {
                'supercategory': label_name,
                'id': idx,
                'name': label_name
            }
            for idx, label_name in self.idx_to_class.items()
        ]
        # Output the annotation json
        ann_dict = {
            'info': info_dict if info_dict is not None else {},
            'licenses': licenses_list if licenses_list is not None else {},
            'images': ann_images,
            'annotations': ann_annotations,
            'categories': ann_categories
        }
        return ann_dict
    
def _output_images_from_dataset(dataset, out_dir, copy_metadata=False):
    os.makedirs(out_dir, exist_ok=True)
    for image_fp in dataset.images:
        if copy_metadata:
            shutil.copy2(image_fp, out_dir)
        else:
            shutil.copy(image_fp, out_dir)

def output_dataset_as_voc(train_dataset: DetectionOutput, val_dataset: DetectionOutput, test_dataset: DetectionOutput, 
                           output_dir, out_train_dir, out_val_dir, out_test_dir):
    """Output the dataset as Pascal VOC format"""
    pass

def _save_coco_annotation(ann_dict, out_dir, ann_filename):
    if ann_dict is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/{ann_filename}', 'w') as f:
            json.dump(ann_dict, f)

def output_dataset_as_coco(output_dir,
                           train_dataset: DetectionOutput, train_ann_name, 
                           val_dataset: DetectionOutput, val_ann_name, 
                           test_dataset: DetectionOutput = None, test_ann_name=None,
                           out_train_dir='train', out_val_dir='val', out_test_dir='test'):
    """Output the dataset as COCO format"""
    # Convert annotation data to coco format
    train_ann = train_dataset.get_coco_annotation()
    val_ann = val_dataset.get_coco_annotation()
    test_ann = test_dataset.get_coco_annotation() if test_dataset is not None else None
    # Save images
    _output_images_from_dataset(train_dataset, f'{output_dir}/{out_train_dir}')
    _output_images_from_dataset(val_dataset, f'{output_dir}/{out_val_dir}')
    if test_dataset is not None:
        _output_images_from_dataset(test_dataset, f'{output_dir}/{out_test_dir}')
    # Save annotation files
    train_ann_filename = train_ann_name if train_ann_name is not None else 'train.json'
    val_ann_filename = val_ann_name if val_ann_name is not None else 'val.json'
    test_ann_filename = test_ann_name if test_ann_name is not None else 'test.json'
    _save_coco_annotation(train_ann, f'{output_dir}/annotations/', train_ann_filename)
    _save_coco_annotation(val_ann, f'{output_dir}/annotations/', val_ann_filename)
    _save_coco_annotation(test_ann, f'{output_dir}/annotations/', test_ann_filename)
