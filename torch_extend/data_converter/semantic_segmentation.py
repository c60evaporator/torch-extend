from pycocotools import mask as coco_mask
import numpy as np
import torch

def convert_polygon_to_mask(segmentations, height, width):
    """
    Convert the polygons to masks
    reference https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
    """
    masks = []
    for polygons in segmentations:
        #polygons = [np.array(polygons).ravel().tolist()]
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = np.any(mask, axis=2).astype(np.uint8)
        masks.append(mask)
    if masks:
        masks = np.stack(masks, axis=0)
    else:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks

def merge_masks(labels: np.ndarray, masks: np.ndarray):
    """
    Merge multiple masks into a mask
    """
    dst_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for label, src_mask in zip(labels, masks):
        dst_mask = np.where((dst_mask == 0) & (src_mask > 0), label, dst_mask)
    return dst_mask

def convert_image_target_to_transformers(image, target, processor, same_img_size,
                                         orig_border_idx=None, orig_bg_idx=None, out_fmt='segformer'):
    """
    Convert image and target from TorchVision to Transformers format (Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)

    Parameters
    ----------
    image : Dict
        Source image data of numpy.ndarray or torch.Tensor with TorchVision format (C, H, W)

    target : Dict
        Source target data with TorchVision format

        torch.Tensor(H, W)

    processor : BaseImageProcessor
        The processor for the Transformers semantic segmentation model

    same_img_size : bool
        Whether all the images have the same size

    orig_border_idx : int
        Border index in the original target data that is replaced with `orig_bg_idx`
    
    orig_bg_idx : int
        Background index in the original target data

    out_fmt : Literal['segformer']
        Format of the output data.
        
        'segformer' is the format for the `SegformerForSemanticSegmentation` model.
    
    Returns
    -------
    item: Dict
        Output data in the Transformers instance segmentation format
        
        segformer: {"pixel_values": torch.Tensor(C, H, W), "labels": torch.Tensor(H, W)}
    """
    if out_fmt == 'segformer':
        if not same_img_size:
            raise ValueError("The images should have the same size for the 'segformer' format.")
        # Replace the border index with the background index if `processor.do_reduce_labels=True`
        if processor.do_reduce_labels:
            if orig_border_idx is None or orig_bg_idx is None:
                raise ValueError("`orig_border_idx` and `orig_bg_idx` should be specified when `processor.do_reduce_labels=True`.")
            bg_mask = torch.full_like(target, orig_bg_idx)
            target = torch.where(target == orig_border_idx, bg_mask, target)
        # Apply the Transformers processor
        encoding = processor(images=image, segmentation_maps=target,
                             return_tensors="pt")
        # Remove batch dimension and return as dictionary
        return {
            "pixel_values": encoding["pixel_values"].squeeze(),
            "labels": encoding["labels"].squeeze()
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

        'transformers': {"pixel_values": torch.Tensor(B, C, H, W), "labels": torch.Tensor(B, H, W)}
    
    Returns
    -------
    images: torch.Tensor(B, C, H, W)
        Images in the batch
    
    targets: torch.Tensor(B, H, W)
        Targets in the batch that indicates the mask with the label indices
    """
    if in_fmt == 'transformers':
        return batch['pixel_values'], batch['labels']
