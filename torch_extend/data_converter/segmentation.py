from pycocotools import mask as coco_mask
import numpy as np

def _convert_polygon_to_mask(segmentations, height, width):
    """
    Convert the polygons to masks
    reference https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
    """
    masks = []
    for polygons in segmentations:
        polygons = [np.array(polygons).ravel().tolist()]
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

def _merge_masks(labels: np.ndarray, masks: np.ndarray):
    """
    Merge multiple masks into a mask
    """
    dst_mask = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
    for label, src_mask in zip(labels, masks):
        dst_mask = np.where((dst_mask == 0) & (src_mask > 0), label, dst_mask)
    return dst_mask