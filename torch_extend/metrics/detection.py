from typing import Dict, List, Literal
from torch import nn, Tensor, no_grad
from torchvision import ops
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import copy
import time

def iou_object_detection(box_pred, label_pred, boxes_true, labels_true,
                         match_label=True):
    """
    Calculate IoU of object detection
    https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/
    """
    # If the label should be matched
    if match_label:
        boxes_gt = []
        labels_gt = []
        for box_true, label_true in zip(boxes_true.cpu().detach(), labels_true.cpu().detach()):
            if label_true == label_pred:
                boxes_gt.append(box_true)
                labels_gt.append(label_true)
    # If the label should NOT be matched
    else:
        boxes_gt = boxes_true.cpu().detach()
        labels_gt = labels_true.cpu().detach()

    # Calculate IoU with every ground truth bbox
    ious = []
    for box_true, label_true in zip(boxes_gt, labels_gt):
        box_true = box_true.view(1, -1)
        box_pred = box_pred.view(1, -1).cpu().detach()
        iou = float(ops.box_iou(box_true, box_pred))
        ious.append(iou)

    # Extract max IoU
    if len(ious) > 0:
        max_iou = max(ious)
    else:
        max_iou = 0.0
    return max_iou


def extract_cofident_boxes(boxes, labels, scores, score_threshold, masks=None):
    """
    Extract bounding boxes whose score > score_threshold

    Parameters
    ----------    
    boxes : array-like of shape (n_boxes, 4)
        A float array of bounding boxes in the format of (xmin, ymin, xmax, ymax).
    
    labels : array-like of shape (n_boxes,)
        An integer array of class labels.

    scores : array-like of shape (n_boxes,)
        A float array of confidence scores.
    
    score_threshold : float
        Bounding boxes whose confidence score exceed this threshold are used as the predicted bounding boxes.
    
    masks : array-like of shape (n_boxes, H, W)
        A float array of masks. This is used for instance segmentation.
    """
    boxes_confident = []
    labels_confident = []
    scores_confident = []
    # Create dummy masks if masks is not set
    masks_confident = []
    if masks is None:
        masks = [None] * len(boxes)
    # Extract bounding boxes whose score > score_threshold
    for score, box, label, mask in zip(scores, boxes.tolist(), labels, masks):
        if score > score_threshold:
            labels_confident.append(label)
            boxes_confident.append(Tensor(box))
            scores_confident.append(score)
            masks_confident.append(mask)
    return boxes_confident, labels_confident, scores_confident, masks_confident

def _get_recall_precision(scores: np.ndarray, corrects: np.ndarray,
                          smoothe: bool=True):
    """
    Calculate precision-recall curve (PR Curve)

    Parameters
    ----------
    scores : array-like of shape (n_boxes,)
        A float array of confidence scores.

    corrects : array-like of shape (n_boxes,)
        A boolean array which indicates whether the bounding box is correct or not.

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.
    """
    # Order by confidence scores
    ordered_indices = np.argsort(-scores)
    corrects_ordered = corrects[ordered_indices]
    # return [0, 0], [0, 1] if the number of correct boxes is zero
    n_boxes = len(corrects_ordered)
    n_trues = corrects.sum()
    if n_trues == 0:
        return np.array([0, 0]), np.array([0, 1])
    # Calculate the precision and the recall
    accumlulated_trues = np.array([np.sum(corrects_ordered[:i+1]) for i in range(n_boxes)])
    precision = accumlulated_trues / np.arange(1, n_boxes + 1)
    recall = accumlulated_trues / n_trues
    # Smooth the precision
    if smoothe:
        precision = np.array([np.max(precision[i:]) for i in range(n_boxes)])
    # to make sure that the precision-recall curve starts at (0, 1)
    recall = np.r_[0, recall]
    precision = np.r_[1, precision]

    return precision, recall

def _average_precision(precision: np.ndarray, recall: np.ndarray,
                       precision_center: bool=False):
    """
    Calculate average precision based on precision and recall scores

    .. note::
        This average precision is based on Area under curve (AUC) AP, NOT based on Interpolated AP. 
        Reference: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

    Parameters
    ----------
    precision : array-like of shape (n_boxes,)
        Precision scores.

    recall : array-like of shape (n_boxes,)
        Recall scores.

    precision_center : bool
        This parameter is used for specifying which value is used as the y value during the calculation of area under curve (AUC).

        If True, use the center value (average of the left and the right value) as the y value like `sklearn.metrics.auc()`

        If False, use the right value as the y value like `sklearn.metrics.average_precision_score()`
    """
    # Calculate AUC (y value calculation method can be changed by `preciision_center` argument)
    if precision_center:
        average_precision = auc(recall, precision)
    else:
        average_precision = np.sum(np.diff(recall) * np.array(precision)[:-1])
    
    return average_precision

def average_precisions(predictions: List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]],
                       targets: List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]],
                       idx_to_class: Dict[int, str],
                       iou_threshold: float=0.5, conf_threshold: float=0.0,
                       smoothe: bool=True, precision_center: bool=False):
    """
    Calculate the average precision of each class label

    .. note::
        This average precision is based on Area under curve (AUC) AP, NOT based on Interpolated AP. 
        Reference: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    
    Parameters
    ----------
    predictions : List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]]
        List of the predicted bounding boxes, labels, and scores

    targets : List[Dict[Literal['boxes', 'labels'], Tensor]]
        List of the true bounding boxes and labels

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
    
    iou_threshold : float
        An IoU threshold that is used for deciding whether the predicted bounding boxes are correct or not.

    conf_threshold : float
        Bounding boxes whose confidence score exceed this threshold are used as the predicted bounding boxes.

        Please set to 0.0 if you follow the general definition of general AP in Object Detection.

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.

    precision_center : bool
        This parameter is used for specifying which value is used as the y value during the calculation of area under curve (AUC).

        If True, use the center value (average of the left and the right value) as the y value like `sklearn.metrics.auc()`

        If False, use the right value as the y value like `sklearn.metrics.average_precision_score()`
    
    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    DEBUG = True

    # Set the default value of conf_threshold
    if conf_threshold is None:
        conf_threshold = 0.0
    # List for storing scores
    labels_pred_all = []
    scores_all = []
    ious_all = []
    correct_all = []
    # For debugging
    image_ids_all = []
    boxes_all = []
    ###### Calculate IoU ######
    # Loop of images
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        # Get predicted bounding boxes
        boxes_pred = prediction['boxes'].cpu().detach()
        labels_pred = prediction['labels'].cpu().detach().numpy()
        scores_pred = prediction['scores'].cpu().detach().numpy()
        # Change the label to -1 if the predicted label is not in idx_to_class
        labels_pred = np.where(np.isin(labels_pred, list(idx_to_class.keys())), labels_pred, -1)
        # Get true bounding boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        # Extract predicted boxes whose score > conf_threshold
        boxes_confident, labels_confident, scores_confident, _ = extract_cofident_boxes(
                boxes_pred, labels_pred, scores_pred, conf_threshold)
        # Calculate IoU
        ious_confident = np.array([
            iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
            for box_pred, label_pred in zip(boxes_confident, labels_confident)
        ])
        # IoU thresholding
        iou_judgement = np.where(ious_confident > iou_threshold, True, False)
        # Store the data
        labels_pred_all.append(np.array(labels_confident))
        scores_all.append(np.array(scores_confident))
        ious_all.append(ious_confident)
        correct_all.append(iou_judgement)
        if i % 500 == 0:  # Show progress every 500 images
            print(f'Calculating IoU: {i}/{len(predictions)} images')
        # For debugging
        if DEBUG:
            image_ids_all.append(np.full(len(labels_pred), i))
            boxes_all.append(boxes_pred.numpy())

    # Concatenate the data
    labels_pred_all = np.concatenate(labels_pred_all)
    scores_all = np.concatenate(scores_all)
    ious_all = np.concatenate(ious_all)
    correct_all = np.concatenate(correct_all)

    # Export prediction and target data to DataFrame for debugging
    if DEBUG:
        image_ids_all = np.concatenate(image_ids_all)
        boxes_all = np.concatenate(boxes_all, axis=0)
        df_preds = pd.DataFrame({"image": image_ids_all,
                                 "box_xmin": boxes_all[:,0], "box_ymin": boxes_all[:,1], "box_xmax": boxes_all[:,2], "box_ymax": boxes_all[:,3],
                                 "label": labels_pred_all,
                                 "score": scores_all,
                                 "iou": ious_all,
                                 "correct": correct_all})
        df_preds.to_csv('preds_all.csv', index=False)
        df_targets = [
            {"image": i, 
             "box_xmin": target_box[0].item(), "box_ymin": target_box[1].item(), "box_xmax": target_box[2].item(), "box_ymax": target_box[3].item(),
             "label": target_label.item()}
            for i, target in enumerate(targets)
            for target_box, target_label in zip(target['boxes'], target['labels'])
        ]
        pd.DataFrame(df_targets).to_csv('target_all.csv', index=False)
    
    ###### Calculate Average Precision ######
    aps = {}
    # Loop of predicted labels
    for label_pred in sorted(set(labels_pred_all)):
        label_name = idx_to_class[label_pred]
        label_indices = np.where(labels_pred_all == label_pred)
        scores_label = scores_all[label_indices]
        correct_label = correct_all[label_indices]
        # Calculate the precision-recall curve (PR Curve)
        precision, recall = _get_recall_precision(scores_label, correct_label, smoothe=smoothe)
        # Calculate the average precision
        average_precision = _average_precision(precision, recall, precision_center=precision_center)
        # Store the result
        aps[label_pred] = {
            'label_name': label_name,
            'average_precision': average_precision,
            'precision': precision,
            'recall': recall
        }
    return aps

def average_precisions_torchvison(dataloader: DataLoader, model: nn.Module, device: Literal['cuda', 'cpu'],
                                  idx_to_class: Dict[int, str],
                                  iou_threshold: float=0.5, conf_threshold: float=0.0,
                                  smoothe: bool=True, precision_center: bool=False):
    """
    Calculate average precisions with TorchVision models and DataLoader

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        List of the predicted bounding boxes

    model : torch.nn.Module
        List of the true bounding boxes

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
    
    iou_threshold : float
        An IoU threshold that is used for deciding whether the predicted bounding boxes are correct or not.

    conf_threshold : float
        Bounding boxes whose confidence score exceed this threshold are used as the predicted bounding boxes.

        Please set to 0.0 if you follow the general definition of general AP in Object Detection.

    precision_center : bool
        This parameter is used for specifying which value is used as the y value during the calculation of area under curve (AUC).

        If True, use the center value (average of the left and the right value) as the y value like `sklearn.metrics.auc()`

        If False, use the right value as the y value like `sklearn.metrics.average_precision_score()`
    
    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    # Predict
    targets_list = []
    predictions_list = []
    start = time.time()  # For elapsed time
    for i, (imgs, targets) in enumerate(dataloader):
        imgs_gpu = [img.to(device) for img in imgs]
        model.eval()  # Set the evaluation mode
        with no_grad():  # Avoid memory overflow
            predictions = model(imgs_gpu)
        # Store the result
        targets_list.extend(targets)
        predictions_cpu = [{k: v.cpu() for k, v in pred.items()} for pred in predictions]
        predictions_list.extend(predictions_cpu)
        if i%100 == 0:  # Show progress every 100 images
            print(f'Prediction for mAP: {i}/{len(dataloader)} batches, elapsed_time: {time.time() - start}')
    aps = average_precisions(predictions_list, targets_list, idx_to_class, iou_threshold, conf_threshold, smoothe, precision_center)
    return aps
