from typing import Dict, Literal, Any
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import math
import numpy as np

from . import detection as det
from ..metrics import detection as det_metrics
from . import semantic_segmentation as semseg

def _convert_masks_to_seg_mask(image, masks, labels, border_mask=None):
    """Convert instance masks to a single segmentation mask."""
    seg_mask = torch.zeros_like(image[0, :, :], dtype=torch.int64)
    accumulated_mask = torch.zeros_like(image[0, :, :], dtype=torch.int64)
    for mask, label in zip(masks, labels):
        seg_mask.masked_fill_(mask.bool(), label)
        accumulated_mask += mask
    # occlusion
    occulded_mask = accumulated_mask > 1
    seg_mask.masked_fill_(occulded_mask, 254)
    # border
    if border_mask is not None:
        seg_mask.masked_fill_(border_mask.bool(), 255)
    return seg_mask

def show_instance_masks(image, masks, boxes=None,
                        border_mask=None,
                        labels=None, idx_to_class=None,
                        colors=None, fill=False, width=1,
                        font=None, font_size=None,
                        alpha=0.4, palette=None, add_legend=True,
                        ax=None):
    """
    Show the image with the masks and bounding boxes for Instance Segmentation.

    Parameters
    ----------
    image : torch.Tensor (C, H, W)
        Input image
    masks : torch.Tensor (N, H, W)
        Target masks of the objects
    boxes : torch.Tensor (N, 4)
        Bounding boxes with Torchvision object detection format
    border_mask : torch.Tensor (H, W)
        Border mask
    labels : torch.Tensor (N)
        Target labels of the objects
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
    alpha : float
        Transparency of the segmentation mask
    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for segmentation mask
    add_legend : bool
        If True, the legend of the segmentation labels is added to the plot
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Show the image with bounding boxes
    if boxes is not None:
        det.show_bounding_boxes(image, boxes, labels=labels, idx_to_class=idx_to_class,
                                colors=colors, fill=fill, width=width,
                                font=font, font_size=font_size, ax=ax)
    else:
        ax.imshow(image.permute(1, 2, 0))
    # Auto palette generation
    if palette is None:
        palette = semseg.create_segmentation_palette()
    # Convert instance masks to single mask
    seg_mask = _convert_masks_to_seg_mask(image, masks, labels, border_mask)
    # Show the mask
    segmentation_img = semseg.array1d_to_pil_image(seg_mask, palette, 
                                                   bg_idx=0, border_idx=255,
                                                   occlusion_idx=254)
    ax.imshow(segmentation_img, alpha=alpha)
    # Add legend
    if add_legend:
        labels_unique = torch.unique(labels).cpu().detach().numpy().tolist()
        # Convert class IDs to class names
        if idx_to_class is None:
            idx_to_class_bd = {idx: str(idx) 
                               for idx in range(np.max(labels_unique))}
        else:
            idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
        # Add the border and occlusion label to idx_to_class
        if 254 in torch.unique(seg_mask).cpu().detach().numpy().tolist():
            labels_unique.append(254)
            idx_to_class_bd[254] = 'occlusion'
        if border_mask is not None:
            labels_unique.append(255)
            idx_to_class_bd[255] = 'border'
        # Make the label text
        label_dict = {k: v for k, v in idx_to_class_bd.items()}
        # Add legend
        handles = [mpatches.Patch(facecolor=(palette[label][0]/255,
                                         palette[label][1]/255,
                                         palette[label][2]/255), 
                              label=label_dict[label])
               for label in labels_unique]
        ax.legend(handles=handles)

def show_predicted_instances(imgs, preds, targets, idx_to_class,
                             border_mask=None,
                             max_displayed_images=10, conf_threshold=0.5,
                             alpha=0.5, palette=None):
    """
    Show images with predicted bounding boxes and masks for Instance Segmentation.

    Parameters
    ----------
    imgs : List[torch.Tensor (C x H x W)]
        List of images that are standardized to [0, 1]
    
    preds : Dict[str, Any] (TorchVision instance segmentation prediction format)
        List of prediction results. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'masks': Tensor([mask1,..]), 'labels': Tensor([labelindex1,..]), 'scores': Tensor([confidence1,..])}]
    
    targets : Dict[str, Any] (TorchVision instance segmentation target format)
        List of the ground truths. The format should be as follows.

        [{'boxes': Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'masks': Tensor([mask1,..]), 'labels': Tensor([labelindex1,..])}]
    
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot

    border_mask : torch.Tensor (H, W)
        Border mask

    max_displayed_images : int
        number of maximum displayed images. This is in case of big batch size.

    conf_threshold : float or List[float]
        threshold of the confidence score for selecting predicted bounding boxes shown or its list.
    
    alpha : float
        Transparency of the segmentation mask

    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for segmentation mask
    """
    # Check if conf_threshold is list or not
    if isinstance(conf_threshold, list):
        conf_thresholds = conf_threshold
    else:
        conf_thresholds = [conf_threshold]
    # image iteration
    for i, (img, pred, target) in enumerate(zip(imgs, preds, targets)):
        img = (img*255).to(torch.uint8).cpu().detach()  # Change from float[0, 1] to uint[0, 255]
        # Prediction
        boxes = pred['boxes'].cpu().detach()
        labels = pred['labels'].cpu().detach().numpy()
        labels = np.where(labels>=len(idx_to_class),-1, labels)  # Modify labels to 0 if the predicted labels are background
        scores = pred['scores'].cpu().detach().numpy()
        masks = pred['masks'].cpu().detach().squeeze(1)
        masks = torch.round(masks).to(torch.uint8)  # float32(N, 1, H, W) -> uint8(N, H, W)
        # Ground truth
        boxes_true = target['boxes']
        labels_true = target['labels']
        masks_true = target['masks']

        # Confidence score threshold iteration
        for conf_thresh in conf_thresholds:
            # Create a camvas
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # Filter out confident bounding boxes whose confidence score > conf_threshold
            if scores is not None:
                boxes_confident, labels_confident, scores_confident, masks_conficent \
                    = det_metrics.extract_cofident_boxes(boxes, labels, scores, conf_thresh,
                                                         masks=masks)
            # Extract all predicted boxes if score is not set
            else:
                boxes_confident = copy.deepcopy(boxes)
                labels_confident = copy.deepcopy(labels)
                scores_confident = None
                masks_conficent = copy.deepcopy(masks)
            # Show Pred bounding boxes whose confidence score > conf_threshold with True boxes
            det._show_pred_true_boxes(img, boxes_confident, labels_confident, 
                                      boxes_true, labels_true,
                                      idx_to_class=idx_to_class,
                                      scores=scores_confident,
                                      calc_iou=True, ax=axes[0])
            axes[0].set_title('Bounding boxes')
            # Show the predicted masks
            show_instance_masks(img, masks_conficent, boxes=None,
                                labels=torch.tensor(labels_confident), idx_to_class=idx_to_class,
                                alpha=alpha, palette=palette,
                                ax=axes[1])
            axes[1].set_title('Predicted masks')
            # Show the ground truth masks
            show_instance_masks(img, masks_true, boxes=None,
                                border_mask=border_mask,
                                labels=labels_true, idx_to_class=idx_to_class,
                                alpha=alpha, palette=palette,
                                ax=axes[2])
            axes[2].set_title('Ground truth masks')
            fig.suptitle(f'Prediction (confident score > {conf_thresh})')
            plt.show()
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break
