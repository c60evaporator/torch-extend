from typing import Dict, Literal, List
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import seaborn as sns
import copy
import math
import numpy as np
from torchmetrics.detection import MeanAveragePrecision

from ..metrics import detection as det

def show_bounding_boxes(image, boxes, labels=None, idx_to_class=None,
                        ious=None, iou_decimal=3,
                        scores=None, score_decimal=3,
                        colors=None, fill=False, width=1,
                        font_size=10, text_color=None,
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
    ious : List[float]
        IoUs of the bounding boxes. If None, IoUs are not displayed.
    iou_decimal : str
        A decimal for the displayed IoUs. Only used when ious is not None.
    scores : List[float]
        Confidence scores for the bounding boxes. If None, the confidence scores are not displayed.
    score_decimal : str
        A decimal for the displayed confidence scores. Only used when scores is not None.
    colors : color or list of colors, optional
        List containing the colors of the boxes or single color for all boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        By default, random colors are generated for boxes.
    fill : bool
        If `True` fills the bounding box with specified color.
    width : int
        Width of the line of the bounding boxes.
    font_size : int
        The requested font size in points.
    text_color : str
        Text color of the label, IoU, and confidence score. If None, the color is determined by `colors` based on the label.
    anomaly_indices : int
        Anomaly box indices displayed as crosses.
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # If ious is None, create dummy ious
    if ious is None:
        ious = torch.full((len(boxes),), float('nan'))
    # If scores is None, create dummy scores
    if scores is None:
        scores = torch.full((len(boxes),), float('nan'))
    # If colors is None, create a color palette
    if colors is None:
        colors = sns.color_palette(n_colors=256).as_hex()

    # Convert class IDs to class names
    if idx_to_class is not None:
        label_names = [idx_to_class[int(label.item())] for label in labels]
    else:
        label_names = [str(label.item()) for label in labels]
    # Show the image
    ax.imshow(image.permute(1, 2, 0))
    # Show bounding boxes
    for box, label, label_name, score, iou in zip(boxes, labels, label_names, scores, ious):
        # Show Rectangle of the bounding box
        r = patches.Rectangle(xy=(box[0].item(), box[1].item()), 
                              width=box[2].item()-box[0].item(), 
                              height=box[3].item()-box[1].item(), 
                              ec=colors[label.item()], fill=fill,
                              lw=width)
        ax.add_patch(r)
        # Show label, score, and IoU
        shown_text = label_name
        if not torch.isnan(score):
            shown_text += f', score={round(float(score.item()),score_decimal)}'
        if not torch.isnan(iou):
            if iou > 0.0:
                shown_text += f', TP, IoU={round(float(iou.item()),iou_decimal)}'
            else:
                shown_text += ', FP'
        ax.text(box[0].item(), box[1].item()-width, shown_text, 
                color=colors[label.item()] if text_color is None else text_color,
                fontsize=font_size)
    
    # image_with_boxes = draw_bounding_boxes(image, boxes, labels=label_names, colors=colors,
    #                                        fill=fill, width=width,
    #                                        font=font, font_size=100)
    # image_with_boxes = image_with_boxes.permute(1, 2, 0)  # Change axis order from (ch, x, y) to (x, y, ch)
    # ax.imshow(image_with_boxes)
    # Draw Anomaly boxes
    if anomaly_indices is not None:
        for idx in anomaly_indices:
            ax.plot(boxes[idx][0].item(), boxes[idx][1].item(), marker='X', markersize=6, color = 'red') #　Anomaly topleft
            plt.text(boxes[idx][0].item(), boxes[idx][1].item()-width, labels[idx].item(), color='red', fontsize=8)
            ax.plot(boxes[idx][2].item(), boxes[idx][3].item(), marker='X', markersize=6, color = 'red') #　Anomaly bottomright

def show_predicted_bboxes(imgs, preds, targets, idx_to_class,
                          max_displayed_images=10, score_threshold=0.2,
                          calc_iou=True, score_decimal=3, iou_decimal=3,
                          colors=None, fill=False, width=1,
                          font_size=10, figsize=(12, 6)) -> List[Figure]:
    """
    Show minibatch images with predicted bounding boxes.

    Parameters
    ----------
    imgs : List[torch.Tensor (C, H, W)] or List[torch.Tensor (B, C, H, W)]
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
    score_threshold : float or List[float]
        A threshold of the confidence score for selecting predicted bounding boxes shown.
    calc_iou : True
        If True, IoUs are calculated and shown
    score_decimal : str
        A decimal for the displayed confidence scores.
    iou_decimal : str
        A decimal for the displayed IoUs.
    colors : color or list of colors, optional
        List containing the colors of the boxes or single color for all boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        By default, random colors are generated for boxes.
    fill : bool
        If `True` fills the bounding box with specified color.
    width : int
        Width of the line of the bounding boxes.
    font_size : int
        The requested font size in points.
    figsize: Tuple[int, int]
        Figure size of the plot.
    """
    # Check if score_threshold is list or not
    if isinstance(score_threshold, list):
        score_thresholds = score_threshold
    else:
        score_thresholds = [score_threshold]
    
    figures = []
    # Iterate over the images
    for i, (img, pred, target) in enumerate(zip(imgs, preds, targets)):
        # Convert the image from float32 [0, 1] to uint8 [0, 255]
        img = (img*255).to(torch.uint8).cpu().detach()
        # Extract the ground truth boxes and labels
        boxes_true = target['boxes'].cpu().detach()
        labels_true = target['labels'].cpu().detach()
        # Extract the predicted boxes, labels, and scores
        boxes = pred['boxes'].cpu().detach()
        labels = pred['labels'].cpu().detach().numpy()
        scores = pred['scores'].cpu().detach().numpy() if 'scores' in pred else None
        # Change the label to -1 if the predicted label is not in idx_to_class
        labels = np.where(np.isin(labels, list(idx_to_class.keys())), labels, -1)
        idx_to_class_uk = {k: v for k, v in idx_to_class.items()}
        idx_to_class_uk[-1] = 'unknown'

        # Confidence score threshold iteration
        for score_thresh in score_thresholds:
            # Create a canvas for plotting
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Show the ground truth boxes
            show_bounding_boxes(img, boxes_true, labels=labels_true, 
                                idx_to_class=idx_to_class,
                                colors=colors, fill=fill, width=width, font_size=font_size,
                                ax=axes[0])
            axes[0].set_title('True bounding boxes')

            # Filter out confident bounding boxes whose confidence score > score_thresh
            if scores is not None:
                boxes_confident, labels_confident, scores_confident, _ = det.extract_cofident_boxes(
                        boxes, labels, scores, score_thresh)
            # Extract all predicted boxes if score is not set
            else:
                boxes_confident = copy.deepcopy(boxes)
                labels_confident = copy.deepcopy(labels)
                scores_confident = None
            # Calculate IoU
            if calc_iou:
                ious = torch.tensor([
                    det.iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
                    for box_pred, label_pred in zip(boxes_confident, labels_confident)
                ], dtype=torch.float32)
            else:
                ious = None
            # Show the predicted bounding boxes with scores and IoUs
            show_bounding_boxes(img, boxes_confident, labels=labels_confident,
                                idx_to_class=idx_to_class, 
                                ious=ious, iou_decimal=iou_decimal,
                                scores=scores_confident, score_decimal=score_decimal,
                                colors=colors, fill=fill, width=width, font_size=font_size,
                                ax=axes[1])
            plt.title(f'Predicted bounding boxes (Score > {score_thresh})')
            plt.show()
            figures.append(fig)
        
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

    return figures

def show_average_precisions(predictions: List[Dict[Literal['boxes', 'labels', 'scores'], torch.Tensor]],
                            targets: List[Dict[Literal['boxes', 'labels', 'scores'], torch.Tensor]],
                            idx_to_class: Dict[int, str],
                            zero_background: bool=True,
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

    zero_background : bool
        If True, the label index of output metrics is reduced by 1 compared to `idx_to_class`. Typically, this is set to `True` when the background class is 0 and other classes are 1, 2, 3, ...

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
        displayed_idx = idx + 1 if zero_background else idx
        # Show each label's PR Curve and average precision
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(np.append(recalls, 1.0), np.append(precisions.tolist(), 0.0))
        ax.set_title(f"{idx_to_class[displayed_idx]}, index={displayed_idx}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.text(1.08, 1.08,
                f"AP={round(ap, score_decimal)}",
                verticalalignment='top', horizontalalignment='right')
        fig.show()
        # Plot PR Curve on the mean average precision graph
        ax_mean.plot(np.append(recalls, 1.0), np.append(precisions.tolist(), 0.0), label=idx_to_class[displayed_idx])

    ax_mean.set_title(f'mean Average Precision (mAP) @IoU={shown_iou}')
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
