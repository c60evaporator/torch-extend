from typing import List, Dict, Literal
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

from ..data_converter.instance_segmentation import convert_instance_masks_to_semantic_mask
from .semantic_segmentation import segmentation_ious

def mask_iou(mask_pred, label_pred, masks_true, labels_true,
             match_label=True):
    """
    Calculate IoU of the predicted mask for instance segmentation.
    """
    # If the label should be matched
    if match_label:
        masks_gt = []
        labels_gt = []
        for mask_true, label_true in zip(masks_true.cpu().detach(), labels_true.cpu().detach()):
            if label_true == label_pred:
                masks_gt.append(mask_true)
                labels_gt.append(label_true)
    # If the label should NOT be matched
    else:
        masks_gt = masks_true.cpu().detach()
        labels_gt = labels_true.cpu().detach()

    # Calculate IoU with every ground truth bbox
    ious = []
    for mask_true in masks_gt:
        mask_true_flatten = mask_true.numpy().flatten()
        mask_pred_flatten = mask_pred.cpu().detach().numpy().flatten()
        confmat = confusion_matrix(mask_true_flatten, mask_pred_flatten)
        tp = confmat[1, 1]
        fn = confmat[1, 0]
        fp = confmat[0, 1]
        iou = tp / (tp + fn + fp)
        ious.append(float(iou))

    # Extract max IoU
    if len(ious) > 0:
        max_iou = max(ious)
    else:
        max_iou = 0.0
    return max_iou

import matplotlib.pyplot as plt

def instance_mean_ious(preds: List[Dict[Literal['masks', 'labels', 'scores'], torch.Tensor]],
                       targets: List[Dict[Literal['masks', 'labels'], torch.Tensor]],
                       idx_to_class: Dict[int, str],
                       bg_idx:int = 0,
                       border_idx:int = None,
                       score_threshold:float = 0.2):
    """
    Calculate the mean IoUs of the predicted masks for instance segmentation.

    Parameters
    ----------
    preds : List[Dict[Literal['masks', 'labels', 'scores'], torch.Tensor]]
        List of the instance segmentation prediction of Torchvision format.

    targets : List[Dict[Literal['masks', 'labels'], torch.Tensor]]
        List of the instance segmentation target of Torchvision format.

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    bg_idx : int
        The index of the background in the target mask.

    border_idx : int
        Index of the border class. The border area in the target image is ignored in the calculation or IoU.

    score_threshold : float
        The threshold of the score for the predictions used for the calculation of mean IoUs.

    Returns
    -------
    tps_batch : np.ndarray
        Number of true positives of each class in the batch
    fps_batch : np.ndarray
        Number of false positives of each class in the batch
    fns_batch : np.ndarray
        Number of false negatives of each class in the batch
    ious_batch : np.ndarray
        IoUs of each class in the batch
    label_indices : List[int]
        The list of label indices
    confmat_batch : np.ndarray
        Confusion matrix of all the pixels in the batch
    """
    # Score thresholding of the predicted data
    preds_confident = [
        {'masks': pred['masks'][pred['scores'] > score_threshold],
         'labels': pred['labels'][pred['scores'] > score_threshold]}
        for pred in preds
    ]
    # Extract semantic masks from instance masks
    pred_semantic_masks = [
        convert_instance_masks_to_semantic_mask(
            pred['masks'], pred['labels'], bg_idx, 
            border_idx=border_idx, border_mask=None,
            add_occlusion=False, occlusion_priority=None)
        for pred in preds_confident
    ]
    target_semantic_masks = [
        convert_instance_masks_to_semantic_mask(
            target['masks'], target['labels'], bg_idx, 
            border_idx=border_idx, border_mask=target['border_mask'] if 'border_mask' in target else None,
            add_occlusion=False, occlusion_priority=None)
        for target in targets
    ]
    # Calculate IoUs
    tps, fps, fns, ious, label_indices, confmat = segmentation_ious(
            pred_semantic_masks, target_semantic_masks,
            idx_to_class, pred_type='label',
            bg_idx=bg_idx, border_idx=border_idx)

    return tps, fps, fns, ious, label_indices, confmat
