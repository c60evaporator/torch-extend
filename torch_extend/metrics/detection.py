from typing import Dict, List, Literal
from torch import nn, Tensor, no_grad
from torchvision import ops
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import numpy as np
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
        for box_true, label_true in zip(boxes_true, labels_true):
            if label_true == label_pred:
                boxes_gt.append(box_true)
                labels_gt.append(label_true)
    # If the label should NOT be matched
    else:
        boxes_gt = copy.deepcopy(boxes_true)
        labels_gt = copy.deepcopy(labels_true)

    # Calculate IoU with every ground truth bbox
    ious = []
    for box_true, label_true in zip(boxes_gt, labels_gt):
        box_true = box_true.view(1, -1)
        box_pred = box_pred.view(1, -1)
        iou = float(ops.box_iou(box_true, box_pred))
        ious.append(iou)

    # Extract max IoU
    if len(ious) > 0:
        max_iou = max(ious)
    else:
        max_iou = 0.0
    return max_iou


def extract_cofident_boxes(scores, boxes, labels, conf_threshold):
    """Extract bounding boxes whose score > conf_threshold"""
    boxes_confident = []
    labels_confident = []
    scores_confident = []
    for score, box, label in zip(scores, boxes.tolist(), labels):
        if score > conf_threshold:
            labels_confident.append(label)
            boxes_confident.append(Tensor(box))
            scores_confident.append(score)
    return boxes_confident, labels_confident, scores_confident

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
                       iou_threshold: float=0.5, conf_threshold: float=0.2,
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

    smoothe : bool
        If True, the precision-recall curve is smoothed to fix the zigzag pattern.

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
    
    iou_threshold : float
        An IoU threshold that is used for deciding whether the predicted bounding boxes are correct or not.

    conf_threshold : float
        Bounding boxes whose confidence score exceed this threshold are used as the predicted bounding boxes.

        If None, all the predicted bounding boxes are used.

    precision_center : bool
        This parameter is used for specifying which value is used as the y value during the calculation of area under curve (AUC).

        If True, use the center value (average of the left and the right value) as the y value like `sklearn.metrics.auc()`

        If False, use the right value as the y value like `sklearn.metrics.average_precision_score()`
    
    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    if conf_threshold is None:
        conf_threshold = 0.0
    # List for storing scores
    labels_pred_all = []
    scores_all = []
    ious_all = []
    correct_all = []
    ###### Calculate IoU ######
    # Loop of images
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        # Get predicted bounding boxes
        boxes_pred = prediction['boxes'].cpu().detach()
        labels_pred = prediction['labels'].cpu().detach().numpy()
        labels_pred = np.where(labels_pred >= max(idx_to_class.keys()),-1, labels_pred)  # Modify labels to 0 if the predicted labels are background
        scores_pred = prediction['scores'].cpu().detach().numpy()
        # Get true bounding boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        # Extract predicted boxes whose score > conf_threshold
        boxes_confident, labels_confident, scores_confident = extract_cofident_boxes(
                scores_pred, boxes_pred, labels_pred, conf_threshold)
        # Calculate IoU
        ious_confident = [
            iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
            for box_pred, label_pred in zip(boxes_confident, labels_confident)
        ]
        # IoU thresholding
        iou_judgement = np.where(np.array(ious_confident) > iou_threshold, True, False).tolist()
        # Store the data on DataFrame
        labels_pred_all.extend(labels_confident)
        scores_all.extend(scores_confident)
        ious_all.extend(ious_confident)
        correct_all.extend(iou_judgement)
        if i % 500 == 0:  # Show progress every 500 images
            print(f'Calculating IoU: {i}/{len(predictions)} images')
    ###### Calculate Average Precision ######
    aps = {}
    # Loop of predicted labels
    for label_pred in sorted(set(labels_pred_all)):
        label_name = idx_to_class[label_pred]
        label_indices = np.where(np.array(labels_pred_all) == label_pred)
        scores_label = np.array(scores_all)[label_indices]
        correct_label = np.array(correct_all)[label_indices]
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
                                  iou_threshold: float=0.5, conf_threshold: float=0.2,
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

        If None, all the predicted bounding boxes are used.

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
    aps = average_precisions(predictions_list, targets_list, idx_to_class, iou_threshold, conf_threshold)
    return aps
