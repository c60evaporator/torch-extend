from typing import Dict, Literal, Any
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import math
import numpy as np

from ..metrics.detection import iou_object_detection, extract_cofident_boxes

def show_bounding_boxes(image, boxes, labels=None, idx_to_class=None,
                        colors=None, fill=False, width=1,
                        font=None, font_size=None,
                        anomaly_indices=None,
                        ax=None):
    """
    Show the image with the segmentation.

    Parameters
    ----------
    image : torch.Tensor (C x H x W)
        Input image
    boxes : torch.Tensor (N, 4)
        Bounding boxes with Torchvision object detection format
    labels : List[str]
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

def show_pred_true_boxes(image, 
                         boxes_pred, labels_pred,
                         boxes_true, labels_true,
                         idx_to_class = None,
                         color_true = 'green', color_pred = 'red', ax=None,
                         scores=None, conf_threshold=0.0, score_decimal=3,
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
        
        If None, the confidence scores are not displayed and conf_threshold is not applied. 
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
        plt.text(box_true[0], box_true[1], label_true, color=color_true, fontsize=8)

    # Extract predicted boxes whose score > conf_threshold
    if scores is not None:
        boxes_confident, labels_confident, scores_confident = extract_cofident_boxes(
                scores, boxes_pred, labels_pred, conf_threshold)
        print(f'Confident boxes={boxes_confident}, labels={labels_confident}')
    # Extract all predicted boxes if score is not set
    else:
        boxes_confident = copy.deepcopy(boxes_pred)
        labels_confident = copy.deepcopy(labels_pred)
        scores_confident = [float('nan')] * len(boxes_pred)
    # Calculate IoU
    if calc_iou:
        ious_confident = [
            iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
            for box_pred, label_pred in zip(boxes_confident, labels_confident)
        ]
    else:
        ious_confident = [float('nan')] * len(boxes_pred)
    # Display predicted boxes
    for box_pred, label_pred, score, iou in zip(boxes_confident, labels_confident, scores_confident, ious_confident):
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
    # Return result
    return boxes_confident, labels_confident, scores_confident, ious_confident

def show_predicted_detection_minibatch(imgs, predictions, targets, idx_to_class,
                                       max_displayed_images=None, conf_threshold=0.5):
    """
    Show predicted minibatch images with bounding boxes.

    Parameters
    ----------
    imgs : List[torch.Tensor (C x H x W)]
        List of the images which are standardized to [0, 1]
    
    predictions : Dict[str, Any] (TorchVision detection prediction format)
        List of the prediction result. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]
    
    targets : Dict[str, Any] (TorchVision detection target format)
        List of the ground truths. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': Tensor([labelindex1,..])}]
    
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot

    max_displayed_images : int
        number of maximum displayed images. This is in case of big batch size.

    conf_threshold : float
        A threshold of the confidence score for selecting predicted bounding boxes shown.
    """
    for i, (img, prediction, target) in enumerate(zip(imgs, predictions, targets)):
        img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
        boxes = prediction['boxes'].cpu().detach()
        labels = prediction['labels'].cpu().detach().numpy()
        labels = np.where(labels>=len(idx_to_class),-1, labels)  # Modify labels to 0 if the predicted labels are background
        scores = prediction['scores'].cpu().detach().numpy()
        print(f'idx={i}')
        print(f'labels={labels}')
        print(f'scores={scores}')
        print(f'boxes={boxes}')
        # Show all bounding boxes
        show_bounding_boxes(img, boxes, labels=labels, idx_to_class=idx_to_class)
        plt.title('All bounding boxes')
        plt.show()
        # Show Pred bounding boxes whose confidence score > conf_threshold with True boxes
        boxes_true = target['boxes']
        labels_true = target['labels']
        boxes_confident, labels_confident, scores_confident, ious_confident = \
            show_pred_true_boxes(img, boxes, labels, boxes_true, labels_true,
                                idx_to_class=idx_to_class,
                                scores=scores, conf_threshold=conf_threshold,
                                calc_iou=True)
        plt.title('Confident bounding boxes')
        plt.show()
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

def show_average_precisions(aps: Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]],
                            score_decimal: int=3):
    """
    Calculate average precisions with TorchVision models and DataLoader

    Parameters
    ----------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Average precisions that is calculated by `average_precisions()` function

    score_decimal : str
        A decimal for the displayed average precision.
    """
    fig_mean, ax_mean = plt.subplots(1, 1, figsize=(8, 8))

    for k, v in aps.items():
        # Show each label's PR Curve and average precision
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(np.append(v['recall'], 1.0), np.append(v['precision'], 0.0))
        ax.set_title(f"{v['label_name']}, index={int(k)}")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.text(1.08, 1.08,
                f"AP={round(v['average_precision'], score_decimal)}",
                verticalalignment='top', horizontalalignment='right')
        fig.show()
        # Plot PR Curve on the mean average precision graph
        ax_mean.plot(np.append(v['recall'], 1.0), np.append(v['precision'], 0.0), label=v['label_name'])

    mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
    ax_mean.set_title('mean Average Precision (mAP)')
    ax_mean.set_xlim(0, 1.1)
    ax_mean.set_ylim(0, 1.1)
    ax_mean.text(1.08, 1.08,
                 f"mAP={round(mean_average_precision, score_decimal)}",
                 verticalalignment='top', horizontalalignment='right')
    ax_mean.legend()
    fig_mean.show()
