from typing import Dict, Literal, List
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import math
import numpy as np
from torchmetrics.detection import MeanAveragePrecision

from ..metrics import detection as det

def show_bounding_boxes(image, boxes, labels=None, idx_to_class=None,
                        colors=None, fill=False, width=1,
                        font=None, font_size=None,
                        anomaly_indices=None,
                        ax=None):
    """
    Show the image with the bounding boxes.

    Parameters
    ----------
    image : torch.Tensor (C, H, W)
        Input image
    boxes : torch.Tensor (N, 4)
        Bounding boxes with Torchvision object detection format
    labels : torch.Tensor (N)
        Target labels of the bounding boxes
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot
    colors : color or list of colors, optional
        List containing the colors of the boxes or single color for all boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        By default, random colors are generated for boxes.
    fill : bool
        If `True` fills the bounding box with specified color.
    width : int
        Width of bounding box.
    font : str
        A filename containing a TrueType font. If the file is not found in this filename, the loader may
        also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
        `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
    font_size : int
        The requested font size in points.
    anomaly_indices : int
        Anomaly box indices displayed as crosses.
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Convert class IDs to class names
    if idx_to_class is not None:
        labels = [idx_to_class[int(label.item())] for label in labels]
    # Show All bounding boxes
    image_with_boxes = draw_bounding_boxes(image, boxes, labels=labels, colors=colors,
                                           fill=fill, width=width,
                                           font=font, font_size=font_size)
    image_with_boxes = image_with_boxes.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    ax.imshow(image_with_boxes)
    # Draw Anomaly boxes
    if anomaly_indices is not None:
        for idx in anomaly_indices:
            ax.plot(boxes[idx][0], boxes[idx][1], marker='X', markersize=6, color = 'red') #　Anomaly topleft
            plt.text(boxes[idx][0], boxes[idx][1], labels[idx], color='red', fontsize=8)
            ax.plot(boxes[idx][2], boxes[idx][3], marker='X', markersize=6, color = 'red') #　Anomaly bottomright

def _show_pred_true_boxes(image, 
                          boxes_pred, labels_pred,
                          boxes_true, labels_true,
                          idx_to_class = None,
                          color_true = 'green', color_pred = 'red', ax=None,
                          scores=None, score_decimal=3,
                          calc_iou=False, iou_decimal=3):
    """
    Show the true bounding boxes and the predicted bounding boxes

    Parameters
    ----------
    image : torch.Tensor (C x H x W)
        Input image
    boxes_pred : torch.Tensor (N_boxes_pred, 4)
        Predicted bounding boxes with Torchvision object detection format
    labels_pred : torch.Tensor (N_boxes_pred)
        Predicted labels of the bounding boxes
    boxes_true : torch.Tensor (N_boxes, 4)
        True bounding boxes with Torchvision object detection format
    labels_true : torch.Tensor (N_boxes)
        True labels of the bounding boxes
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot
    color_true : str (color name)
        A color for the ture bounding boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
    color_pred : str (color name)
        A color for the predicted bounding boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
    scores : torch.Tensor (N_boxes_pred)
        Confidence scores for the predicted bounding boxes.
        
        If None, the confidence scores are not displayed. 
    conf_threshold : float
        A threshold of the confidence score for selecting predicted bounding boxes shown.
    score_decimal : str
        A decimal for the displayed confidence scores.
    calc_iou : True
        If True, IoUs are calculated and shown
    iou_decimal : str
        A decimal for the displayed IoUs.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # If scores is None, create dummy scores
    if scores is None:
        scores = [float('nan')] * len(boxes_pred)
    # Convert class IDs to class names
    if idx_to_class is not None:
        labels_pred = [idx_to_class[label.item()] for label in labels_pred]
        labels_true = [idx_to_class[label.item()] for label in labels_true]

    # Display raw image
    img_permuted = image.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    ax.imshow(img_permuted)

    # Display true boxes
    for box_true, label_true in zip(boxes_true.tolist(), labels_true):
        r = patches.Rectangle(xy=(box_true[0], box_true[1]), 
                              width=box_true[2]-box_true[0], 
                              height=box_true[3]-box_true[1], 
                              ec=color_true, fill=False)
        ax.add_patch(r)
        ax.text(box_true[0], box_true[1], label_true, color=color_true, fontsize=8)

    # Calculate IoU
    if calc_iou:
        ious_confident = [
            det.iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
            for box_pred, label_pred in zip(boxes_pred, labels_pred)
        ]
    else:
        ious_confident = [float('nan')] * len(boxes_pred)
    # Display predicted boxes
    for box_pred, label_pred, score, iou in zip(boxes_pred, labels_pred, scores, ious_confident):
        # Show Rectangle
        r = patches.Rectangle(xy=(box_pred[0], box_pred[1]), 
                              width=box_pred[2]-box_pred[0], 
                              height=box_pred[3]-box_pred[1], 
                              ec=color_pred, fill=False)
        ax.add_patch(r)
        # Show label, score, and IoU
        text_pred = label_pred if isinstance(label_pred, str) else str(int(label_pred))
        if not math.isnan(score):
            text_pred += f', score={round(float(score),score_decimal)}'
        if calc_iou:
            if iou > 0.0:
                text_pred += f', TP, IoU={round(float(iou),iou_decimal)}'
            else:
                text_pred += ', FP'
        ax.text(box_pred[0], box_pred[1], text_pred, color=color_pred, fontsize=8)

def show_predicted_bboxes(imgs, preds, targets, idx_to_class,
                          max_displayed_images=10, conf_threshold=0.5):
    """
    Show minibatch images with predicted bounding boxes.

    Parameters
    ----------
    imgs : List[torch.Tensor (C x H x W)]
        List of images that are standardized to [0, 1]
    
    preds : Dict[str, Any] (TorchVision detection prediction format)
        List of prediction results. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]
    
    targets : Dict[str, Any] (TorchVision detection target format)
        List of the ground truths. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..])}]
    
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    max_displayed_images : int
        number of maximum displayed images. This is in case of big batch size.

    conf_threshold : float
        A threshold of the confidence score for selecting predicted bounding boxes shown.
    """
    for i, (img, pred, target) in enumerate(zip(imgs, preds, targets)):
        img = (img*255).to(torch.uint8).cpu().detach()  # Change from float[0, 1] to uint[0, 255]
        boxes = pred['boxes'].cpu().detach()
        labels = pred['labels'].cpu().detach().numpy()
        scores = pred['scores'].cpu().detach().numpy() if 'scores' in pred else None
        # Change the label to -1 if the predicted label is not in idx_to_class
        labels = np.where(np.isin(labels, list(idx_to_class.keys())), labels, -1)
        idx_to_class_uk = {k: v for k, v in idx_to_class.items()}
        idx_to_class_uk[-1] = 'unknown'
        # Show all bounding boxes
        show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class_uk)
        plt.title('All bounding boxes')
        plt.show()
        # Filter out confident bounding boxes whose confidence score > conf_threshold
        if scores is not None:
            boxes_confident, labels_confident, scores_confident, _ = det.extract_cofident_boxes(
                    boxes, labels, scores, conf_threshold)
        # Extract all predicted boxes if score is not set
        else:
            boxes_confident = copy.deepcopy(boxes)
            labels_confident = copy.deepcopy(labels)
            scores_confident = None
        # Show the confident bounding boxes with True boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        _show_pred_true_boxes(img, boxes_confident, labels_confident, 
                              boxes_true, labels_true,
                              idx_to_class=idx_to_class_uk,
                              scores=scores_confident,
                              calc_iou=True)
        plt.title(f'Confident bounding boxes (confident score > {conf_threshold})')
        plt.show()
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

def show_average_precisions(predictions: List[Dict[Literal['boxes', 'labels', 'scores'], torch.Tensor]],
                            targets: List[Dict[Literal['boxes', 'labels', 'scores'], torch.Tensor]],
                            idx_to_class: Dict[int, str],
                            iou_thresholds: List[float]=None,
                            shown_iou: float=0.5,
                            rec_thresholds: List[float]=None,
                            score_decimal: int=3):
    """
    Calculate average precisions with TorchVision models and DataLoader

    Parameters
    ----------
    predictions : List[Dict[Literal['boxes', 'labels', 'scores'], Tensor]]
        List of the predicted bounding boxes, labels, and scores

    targets : List[Dict[Literal['boxes', 'labels'], Tensor]]
        List of the true bounding boxes and labels

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    iou_thresholds : List[float]
        IoU thresholds for evaluation. If set to `None` it corresponds to the stepped range `[0.5,...,0.95]` with step `0.05`. Else provide a list of floats.
    
    shown_iou: float
        An IoU threshold for displaying the precision-recall curve.

    rec_thresholds : List[float]
        Recall thresholds for evaluation. If set to `None` it corresponds to the stepped range `[0.0,...,1.0]` with step `0.01`. Else provide a list of floats.

    score_decimal : str
        A decimal for the displayed average precision.
    """
    # Calculate average precisions by TorchMetrics
    map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, extended_summary=True,
                                      iou_thresholds=iou_thresholds, rec_thresholds=rec_thresholds)
    map_metric.update(predictions, targets)
    map_score = map_metric.compute()
    # Validation of iou_thresholds
    if shown_iou not in map_metric.iou_thresholds:
        raise ValueError(f"The `shown_iou` argument should be in `iou_thresholds`")
    shown_idx = map_metric.iou_thresholds.index(shown_iou)
    # Extract precision curve
    class_precisions = map_score["precision"].numpy()[shown_idx,:,:,0,-1].T
    recalls = map_metric.rec_thresholds
    # Extract average precisions
    aps = np.mean(map_score["precision"].numpy(), axis=1)
    iou_class_aps = aps[:,:,0,-1]
    class_aps = iou_class_aps[shown_idx,:]
    mean_average_precision = np.mean(class_aps)

    fig_mean, ax_mean = plt.subplots(1, 1, figsize=(8, 8))

    for idx, (ap, precisions) in enumerate(zip(class_aps, class_precisions)):
        # Show each label's PR Curve and average precision
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(np.append(recalls, 1.0), np.append(precisions.tolist(), 0.0))
        ax.set_title(f"{idx_to_class[idx]}, index={idx}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.text(1.08, 1.08,
                f"AP={round(ap, score_decimal)}",
                verticalalignment='top', horizontalalignment='right')
        fig.show()
        # Plot PR Curve on the mean average precision graph
        ax_mean.plot(np.append(recalls, 1.0), np.append(precisions.tolist(), 0.0), label=idx_to_class[idx])

    ax_mean.set_title(f'mean Average Precision (mAP) @{shown_iou}')
    ax_mean.set_xlim(0, 1.1)
    ax_mean.set_ylim(0, 1.1)
    ax_mean.text(1.08, 1.08,
                 f"mAP@{shown_iou}={round(mean_average_precision, score_decimal)}",
                 verticalalignment='top', horizontalalignment='right')
    ax_mean.legend()
    fig_mean.show()

# def show_average_precisions(aps: Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]],
#                             score_decimal: int=3):
#     """
#     Calculate average precisions with TorchVision models and DataLoader

#     Parameters
#     ----------
#     aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
#         Average precisions that is calculated by `average_precisions()` function

#     score_decimal : str
#         A decimal for the displayed average precision.
#     """
#     fig_mean, ax_mean = plt.subplots(1, 1, figsize=(8, 8))

#     for k, v in aps.items():
#         # Show each label's PR Curve and average precision
#         fig, ax = plt.subplots(1, 1, figsize=(4, 4))
#         ax.plot(np.append(v['recall'], 1.0), np.append(v['precision'], 0.0))
#         ax.set_title(f"{v['label_name']}, index={int(k)}")
#         ax.set_xlim(0, 1.1)
#         ax.set_ylim(0, 1.1)
#         ax.text(1.08, 1.08,
#                 f"AP={round(v['average_precision'], score_decimal)}",
#                 verticalalignment='top', horizontalalignment='right')
#         fig.show()
#         # Plot PR Curve on the mean average precision graph
#         ax_mean.plot(np.append(v['recall'], 1.0), np.append(v['precision'], 0.0), label=v['label_name'])

#     mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
#     ax_mean.set_title('mean Average Precision (mAP)')
#     ax_mean.set_xlim(0, 1.1)
#     ax_mean.set_ylim(0, 1.1)
#     ax_mean.text(1.08, 1.08,
#                  f"mAP={round(mean_average_precision, score_decimal)}",
#                  verticalalignment='top', horizontalalignment='right')
#     ax_mean.legend()
#     fig_mean.show()
