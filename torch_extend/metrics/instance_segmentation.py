from sklearn.metrics import confusion_matrix
import numpy as np

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
