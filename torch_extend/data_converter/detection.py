from typing import Literal
import torch
from torchvision.ops import box_convert

def _convert_bbox_xywh_to_xyxy(x_min, y_min, w, h):
    """Convert [x_min, y_min, w, h] (COCO) to [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision)"""
    return [x_min, y_min, x_min + w, y_min + h]

def _convert_bbox_xyxy_to_xywh(x_min, y_min, x_max, y_max):
    """Convert [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision) to [x_min, y_min, w, h] (COCO)"""
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def _convert_bbox_centerxywh_to_xyxy(x_c, y_c, w, h):
    """Convert [x_c, y_c, w, h] (YOLO) to [x_min, y_min, x_max, y_max] (PascalVOC & TorchVision)"""
    return [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]

def _convert_bbox_xyxy_to_centerxywh(x_min, y_min, x_max, y_max):
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

def convert_target(target, in_fmt, out_fmt, class_to_idx=None,
                   img_height=None, img_width=None):
    """
    Convert the format of target data

    Parameters
    ----------
    target : {dict, list}
        Input target data (dict or list that includes bounding boxes and labels)

    in_fmt : Literal['torchvision', 'pascal_voc', 'coco', 'yolo']
        Annotation format of the input target data.

        'torchvision': {'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1, labelindex2,..])}

        'pascal_voc': {'annotation': {'object': [{'bndbox': {'xmin':xmin1, 'ymin':ymin1, 'xmax':xmax1, 'ymax':ymax1}, 'name': labelname1},..]}}

        'coco': [{'bbox': [xmin1, ymin1, w1, h1], 'category_id': labelindex1,..},..]

        'yolo': [[labelindex1, nxcenter1, nycenter1, w1, h1], [labelindex2, nycenter2,..],..]]

    out_fmt : Literal['torchvision', 'pascal_voc', 'coco', 'yolo']
        Annotation format of the output target data.
    
    class_to_idx : dict
        A dict which convert class name to class id. Only necessary for 'pascal_voc' format.

    img_height : int
        The height of the image. Only available if ``out_fmt="yolo"``.

    img_width : int
        The width of the image. Only available if ``out_fmt="yolo"``.
    """
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    ###### Convert the target to TorchVision object detection format ######
    # From TorchVision format
    if in_fmt == 'torchvision':
        boxes = target['boxes']
        labels = target['labels']
    # From Pascal VOC Object detection format
    if in_fmt == 'pascal_voc':
        objects = target['annotation']['object']
        box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
        boxes = [[int(obj['bndbox'][k]) for k in box_keys] for obj in objects]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [class_to_idx[obj['name']] for obj in objects]
        labels = torch.tensor(labels)
    # From COCO format
    elif in_fmt == 'coco':
        boxes = [[int(k) for k in _convert_bbox_xywh_to_xyxy(*obj['bbox'])]
             for obj in target]
        boxes = torch.tensor(boxes)
        # Get labels
        labels = [obj['category_id'] for obj in target]
        labels = torch.tensor(labels)
    
    ###### Convert the target from TorchVision format to the specified format ######
    # To TorchVision format
    if out_fmt == 'torchvision':
        converted_target = {'boxes': boxes, 'labels': labels}
    # To Pascal VOC Object detection format
    if out_fmt == 'pascal_voc':
        labels = [idx_to_class[label] for label in labels.tolist()]
        converted_target = {}
        converted_target['annotation'] = {}
        converted_target['annotation']['object'] = [
            {
                'bndbox': {'xmin':box[0], 'ymin':box[1], 'xmax':box[2], 'ymax':box[3]},
                'name': label
            }
            for box, label in zip(boxes.tolist(), labels)
        ]
    # To COCO format
    elif out_fmt == 'coco':
        boxes = [[int(k) for k in _convert_bbox_xyxy_to_xywh(*box)] for box in boxes.tolist()]
        converted_target = [
            {
                'bbox': box,
                'category_id': label
            }
            for  box, label in zip(boxes, labels.tolist())
        ]
    # To YOLO format
    elif out_fmt == 'yolo':
        boxes = [_convert_bbox_xyxy_to_centerxywh(*box) for box in boxes.tolist()]
        boxes = [[box[0]/img_width, box[1]/img_height, box[2]/img_width, box[3]/img_height] for box in boxes]
        converted_target = [
            [label, box[0], box[1], box[2], box[3]]
            for box, label in zip(boxes, labels.tolist())
        ]
    return converted_target

def convert_image_target_to_transformers(image, target, image_id, processor, out_fmt='detr'):
    """
    Convert image and target from TorchVision to Transformers format (Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)

    Parameters
    ----------
    image : Dict
        Source image data with TorchVision format torch.Tensor(C, H, W)

    target : Dict
        Source target data with TorchVision format

        {'boxes': torch.Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': torch.Tensor([labelindex1, labelindex2,..])}

    image_id : int
        Image ID

    processor : BaseImageProcessor
        The processor for the Transformers object detection model

    out_fmt : Literal['detr']
        Format of the output data. 
        
        'detr' is the format for the `DetrForObjectDetection` model.
    
    Returns
    -------
    item: Dict
        Output data in the Transformers object detection format
        
        detr: {"pixel_values": torch.Tensor(C, H, W), "pixel_mask": torch.Tensor(H, W), "labels": {"boxes": torch.Tensor(n_instances, 4), "class_labels": torch.Tensor(n_instances), "org_size": Tuple[int, int],...}} with boxes in normalized cxcywh format.
    """
    if out_fmt == 'detr':
        # format annotations in COCO format
        annotations = [
            {
                "image_id": image_id,
                "category_id": label.item(),
                "bbox": box_convert(box, "xyxy", "xywh").tolist(),
                "iscrowd": 0,
                "area": (box[2].item() - box[0].item()) * (box[3].item() - box[1].item()),
            }
            for box, label in zip(target['boxes'], target['labels'])
        ]
        # Apply the Transformers processor
        encoding = processor(images=image,
                            annotations={'image_id': image_id, 'annotations': annotations},
                            return_tensors="pt")
        # Remove batch dimension and return as dictionary
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "pixel_mask": encoding["pixel_mask"].squeeze(),
            "labels": encoding["labels"][0]
        }

def convert_batch_to_torchvision(batch, in_fmt='transformers'):
    """
    Convert the batch to the torchvision format images and targets

    Parameters
    ----------
    batch : Dict
        Source batch data (transformers object detection format)

    in_fmt : Literal['transformers']
        Format of the input batch data.

        'transformers': {"pixel_values": torch.Tensor(B, C, H, W), "pixel_mask": torch.Tensor(B, H, W), "labels": [{"boxes": torch.Tensor(n_instances, 4), "class_labels": torch.Tensor(n_instances), "org_size": Tuple[int, int],...},...]} with boxes in normalized cxcywh format
    
    Returns
    -------
    images: List[torch.Tensor(C, H, W)]
        Images in the batch
    
    targets: List[{{"boxes": torch.Tensor(n_instances, 4), "labels": torch.Tensor(n_instances)}]
        Targets in the batch

        The box format is [xmin, ymin, xmax, ymax]
    """
    images = []
    targets = []
    if in_fmt == 'transformers':
        device = batch['pixel_values'].device
        for pixel_value, pixel_mask, labels in zip(batch['pixel_values'], batch['pixel_mask'], batch['labels']):
            # Get the mask rectangle
            nonzero_rows = torch.nonzero(pixel_mask.sum(dim=1))
            y_min, y_max = nonzero_rows[0].item(), nonzero_rows[-1].item()
            nonzero_cols = torch.nonzero(pixel_mask.sum(dim=0))
            x_min, x_max = nonzero_cols[0].item(), nonzero_cols[-1].item()
            # Extract the effective area
            image = pixel_value[:, y_min:y_max+1, x_min:x_max+1]
            w, h = x_max - x_min + 1, y_max - y_min + 1
            # Convert the labels to the target
            target = {}
            boxes = box_convert(labels['boxes'], "cxcywh", "xyxy")
            target['boxes'] = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=device) if boxes.shape[0] > 0 \
                    else torch.zeros(size=(0, 4), dtype=torch.float32, device=device)
            target['labels'] = labels['class_labels']
            
            images.append(image)
            targets.append(target)
    
    return images, targets
