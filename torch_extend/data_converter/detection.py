import torch
import os

class DetectionOutput():
    def _get_images_targets(self):
        sample_size = self.__len__()
        images_targets = [self.__getitem__(idx) for idx in range(sample_size)]
        images = [image for image, target in images_targets]
        targets = [target for image, target in images_targets]
        return images, targets

    def output_coco_annotation(self, info_dict=None, licenses_list=None):
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
        # Create "annotations" field in the annotation file
        ann_annotations = []
        obj_id_cnt = 0
        for i_img, target in enumerate(targets):
            for obj in target:
                obj_ann = {
                    'segmentation': [],
                    'area': 0,
                    'iscrowd': 0,
                    'image_id': i_img,
                    'bbox': obj['bbox'],
                    'category_id': obj['category_id'],
                    'id': obj_id_cnt
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

def convert_bbox_xywh_to_xyxy(x_min, y_min, w, h):
    """Convert [x_min, y_min, w, h] (COCO) to [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision)"""
    return [x_min, y_min, x_min + w, y_min + h]

def convert_bbox_xyxy_to_xywh(x_min, y_min, x_max, y_max):
    """Convert [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision) to [x_min, y_min, w, h] (COCO)"""
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def convert_bbox_centerxywh_to_xyxy(x_c, y_c, w, h):
    """Convert [x_c, y_c, w, h] (YOLO) to [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision)"""
    return [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]

def convert_bbox_xyxy_to_centerxywh(x_min, y_min, x_max, y_max):
    """Convert [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision) to [x_c, y_c, w, h] (YOLO)"""
    return [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]

def resize_target(target, src_image, resized_image):
    """
    Resize the target in accordance with the image Resize

    Parameters
    ----------
    target : Dict
        Target data (Torchvision object detection format)

        {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

    src_image : Tensor
        An image before resize (h, w)
    
    resized_image : Tensor
        An image after resize (h, w)
    """
    src_size = src_image.size()
    resized_size = resized_image.size()
    h_ratio = resized_size[1] / src_size[1]
    w_ratio = resized_size[2] / src_size[2]

    targets_resize = {
        'boxes': target['boxes'] * torch.tensor([w_ratio, h_ratio, w_ratio, h_ratio], dtype=float),
        'labels': target['labels']
        }

    return targets_resize

def target_transform_to_torchvision(target, in_format, class_to_idx=None):
    """
    Transform target data to adopt to TorchVision object detection format below

    {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

    This function is only compatible with torchvision.datasets.VOCDetection and torchvision.datasets.COCODetection dataset class.

    Parameters
    ----------
    target : {dict, list}
        Input target data (dict or list that includes bounding boxes and labels)

    in_format : {'pascal_voc', 'coco', 'yolo'}
        Annotation format of the input target data.

        'pascal_voc': {'annotation': {'object': [{'bndbox': {'xmin':xmin1, 'ymin':ymin1, 'xmax':xmax1, 'ymax':ymax1}, 'name': labelname1},..]}}

        'coco': [{'bbox': [xmin1, ymin1, w1, h1], 'category_id': labelindex1,..},..]
    
    class_to_idx : dict
        A dict which convert class name to class id. Only necessary for 'pascal_voc' format.
    """
    # From Pascal VOC Object detection format
    if in_format == 'pascal_voc':
        objects = target['annotation']['object']
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [class_to_idx[obj['name']] for obj in objects]
        labels = torch.tensor(labels)
    # From COCO format
    elif in_format == 'coco':
        boxes = [[int(k) for k in convert_bbox_xywh_to_xyxy(*obj['bbox'])]
             for obj in target]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [obj['category_id'] for obj in target]
        labels = torch.tensor(labels)
    # Make a target dict whose keys are 'boxes' and 'labels'
    target = {'boxes': boxes, 'labels': labels}
    return target

def target_transform_from_torchvision(target, out_format, idx_to_class=None, img_height=None, img_width=None):
    """
    Transform target data from TorchVision object detection format below

    {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

    Parameters
    ----------
    target : {dict, list}
        Input target data (dict or list that includes bounding boxes and labels)

    out_format : {'pascal_voc', 'coco', 'yolo'}
        Annotation format of the input target data.

        'pascal_voc': {'annotation': {'object': [{'bndbox': {'xmin':xmin1, 'ymin':ymin1, 'xmax':xmax1, 'ymax':ymax1}, 'name': labelname1},..]}}

        'coco': [{'bbox': [xmin1, ymin1, w1, h1], 'category_id': labelindex1,..},..]

        'yolo': [[labelindex1, nxcenter1, nycenter1, w1, h1], [labelindex2, nycenter2,..],..]]
    
    idx_to_class : dict
        A dict which convert class id to class name. Only necessary for 'pascal_voc' format.

    img_height : int
        The height of the image. Only necessary for 'yolo' format.

    img_width : int
        The width of the image. Only necessary for 'yolo' format.
    """
    boxes = target['boxes'].tolist()
    labels = target['labels'].tolist()
    # To Pascal VOC Object detection format
    if out_format == 'pascal_voc':
        labels = [idx_to_class[label] for label in labels]
        converted_target = {}
        converted_target['annotation'] = {}
        converted_target['annotation']['object'] = [
            {
                'bndbox': {'xmin':box[0], 'ymin':box[1], 'xmax':box[2], 'ymax':box[3]},
                'name': label
            }
            for box, label in zip(boxes, labels)
        ]
    # To COCO format
    elif out_format == 'coco':
        boxes = [[int(k) for k in convert_bbox_xyxy_to_xywh(*box)] for box in boxes]
        converted_target = [
            {
                'bbox': box,
                'category_id': label
            }
            for  box, label in zip(boxes, labels)
        ]
    # To YOLO format
    elif out_format == 'yolo':
        boxes = [convert_bbox_xyxy_to_centerxywh(*box) for box in boxes]
        boxes = [[box[0]/img_width, box[1]/img_height, box[2]/img_width, box[3]/img_height] for box in boxes]
        converted_target = [
            [label, box[0], box[1], box[2], box[3]]
            for box, label in zip(boxes, labels)
        ]
    return converted_target
