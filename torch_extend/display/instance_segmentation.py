from typing import List, Literal, Any, Dict
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
import copy
import numpy as np

from . import detection as det
from . import semantic_segmentation as semseg
from ..metrics import detection as det_metrics
from ..metrics import instance_segmentation as inst_metrics
from ..data_converter.instance_segmentation import convert_instance_masks_to_semantic_mask
from torch_extend.display.semantic_segmentation import show_predicted_segmentations

def show_instance_masks(image, masks, boxes=None,
                        border_mask=None,
                        labels=None, idx_to_class=None,
                        bg_idx=0, border_idx=None,
                        mask_ious=None, box_ious=None, iou_decimal=3,
                        scores=None, score_decimal=3,
                        colors=None, fill=False, width=1,
                        font_size=10, text_color=None,
                        alpha=0.4, add_legend=True,
                        bg_color=[255, 255, 255], border_color=[0, 0, 0], occlusion_color=[64, 64, 64],
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
    bg_idx : int
        The index of the background in the target mask.
    border_idx : int
        The index of the border in the target mask.
    mask_ious : List[float]
        IoUs of the masks. If None, IoUs are not displayed.
    box_ious : List[float]
        IoUs of the box_ious. If None, IoUs are not displayed.
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
        Width of bounding box.
    font_size : int
        The requested font size in points.
    text_color : str
        Text color of the label, IoU, and confidence score. If None, the color is determined by `colors` based on the label.
    alpha : float
        Transparency of the segmentation mask
    add_legend : bool
        If True, the legend of the segmentation labels is added to the plot
    bg_color : List[int]
        The color of the background described in [R, G, B]
    border_color : List[int]
        The color of the border described in [R, G, B]
    occlusion_color : List[int]
        The color of the occlusion described in [R, G, B]
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    """
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # If colors is None, create a color palette
    if colors is None:
        colors = sns.color_palette(n_colors=256).as_hex()
    
    # Show the image with bounding boxes
    if boxes is not None:
        det.show_bounding_boxes(image, boxes, labels=labels, idx_to_class=idx_to_class,
                                scores=scores, score_decimal=score_decimal,
                                ious=box_ious, iou_decimal=iou_decimal,
                                colors=colors, fill=fill, width=width,
                                font_size=font_size, text_color=text_color,
                                ax=ax)
    else:
        ax.imshow(image.permute(1, 2, 0))
        # Create dummy boxes from the masks
        nonzero_masks = [mask.nonzero() for mask in masks]
        boxes = torch.tensor([
            [nonzero[:, 1].min(), nonzero[:, 0].min(),
             nonzero[:, 1].max(), nonzero[:, 0].max()]
            for nonzero in nonzero_masks
        ])
    
    # Generate a color palette for the segmentation mask
    palette = semseg.create_segmentation_palette(colors)
    # Convert instance masks to single mask
    seg_mask = convert_instance_masks_to_semantic_mask(masks, labels, bg_idx, border_idx, border_mask,
                                                       add_occlusion=True, occlusion_priority=None)
    # Show the mask
    segmentation_img = semseg.array1d_to_pil_image(seg_mask, palette,
                                                   bg_idx=bg_idx, border_idx=border_idx,
                                                   occlusion_idx=254,
                                                   bg_color=bg_color, border_color=border_color,
                                                   occlusion_color=occlusion_color)
    ax.imshow(segmentation_img, alpha=alpha)
    # Show IoUs
    if mask_ious is not None:
        for mask_iou, box, label in zip(mask_ious, boxes, labels):
            # Add text
            ax.text(box[0].item(), box[1].item()+int(font_size*1.5)+width,
                    f'maskIoU={mask_iou:.{iou_decimal}f}',
                    color=colors[label.item()] if text_color is None else text_color,
                    fontsize=font_size)

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
                             bg_idx=0, border_idx=None,
                             border_mask=None, separate_boxes=False,
                             max_displayed_images=10, score_threshold=0.2,
                             calc_iou=True, score_decimal=3, iou_decimal=3,
                             colors=None, fill=False, width=1,
                             font_size=10, text_color=None, figsize=None,
                             alpha=0.5) -> List[Figure]:
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
    bg_idx : int
        The index of the background in the target mask.
    border_idx : int
        The index of the border in the target mask.
    border_mask : torch.Tensor (H, W)
        Border mask
    separate_boxes : bool
        If True, the bounding boxes are shown separately for each image.
    max_displayed_images : int
        number of maximum displayed images. This is in case of big batch size.
    score_threshold : float or List[float]
        threshold of the confidence score for selecting predicted bounding boxes shown or its list.
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
    text_color : str
        Text color of the label, IoU, and confidence score. If None, the color is determined by `colors` based on the label.
    figsize: Tuple[int, int]
        Figure size of the plot. If `None`, the size is determined based on the image aspect ratio.
    alpha : float
        Transparency of the segmentation mask
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
        # Extract the ground truth boxes, masks, and labels
        boxes_true = target['boxes'].cpu().detach()
        labels_true = target['labels'].cpu().detach()
        masks_true = target['masks'].cpu().detach()
        # Extract the predicted boxes, masks, and labels
        boxes = pred['boxes'].cpu().detach()
        labels = pred['labels'].cpu().detach().numpy()
        scores = pred['scores'].cpu().detach().numpy() if 'scores' in pred else None
        masks = pred['masks'].cpu().detach()
        # Mask float32(N, 1, H, W) -> uint8(N, H, W)
        if masks.dtype == torch.float32:
            masks = torch.round(masks.squeeze(1)).to(torch.uint8)
        # Change the label to -1 if the predicted label is not in idx_to_class
        labels = np.where(np.isin(labels, list(idx_to_class.keys())), labels, -1)
        idx_to_class_uk = {k: v for k, v in idx_to_class.items()}
        idx_to_class_uk[-1] = 'unknown'
        
        # Confidence score threshold iteration
        for score_thresh in score_thresholds:
            # If figsize is None, set the size based on the image aspect ratio
            if figsize is None:
                aspect_ratio = img.size(1) / img.size(2)
                figsize = (12, 12*aspect_ratio) if separate_boxes else (12, 6*aspect_ratio)
            
            # Create a canvas for plotting
            if separate_boxes:
                fig, axes = plt.subplots(2, 2, figsize=figsize)
            else:
                fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Show the ground truth masks and boxes
            show_instance_masks(img, masks_true, 
                                boxes=None if separate_boxes else boxes_true,
                                border_mask=border_mask,
                                labels=labels_true, idx_to_class=idx_to_class_uk,
                                bg_idx=bg_idx, border_idx=border_idx,
                                colors=colors, fill=fill, width=width,
                                font_size=font_size, text_color=text_color, alpha=alpha,
                                ax=axes[0][0] if separate_boxes else axes[0])
            if separate_boxes:
                det.show_bounding_boxes(img, boxes_true,
                                        labels=labels, idx_to_class=idx_to_class,
                                        colors=colors, fill=fill, width=width,
                                        font_size=font_size, text_color=text_color,
                                        ax=axes[1][0])
                axes[0][0].set_title('Ground truth masks')
                axes[1][0].set_title('Ground truth boxes')
            else:
                axes[0].set_title('Ground truth masks and boxes')

            # Filter out confident bounding boxes whose confidence score > conf_threshold
            if scores is not None:
                boxes_confident, labels_confident, scores_confident, masks_conficent \
                    = det_metrics.extract_cofident_boxes(boxes, labels, scores, score_thresh,
                                                         masks=masks)
            # Extract all predicted boxes if score is not set
            else:
                boxes_confident = copy.deepcopy(boxes)
                labels_confident = copy.deepcopy(labels)
                scores_confident = None
                masks_conficent = copy.deepcopy(masks)
            # Calculate IoUs
            if calc_iou:
                box_ious = [
                    det_metrics.iou_object_detection(box_pred, label_pred, boxes_true, labels_true)
                    for box_pred, label_pred in zip(boxes_confident, labels_confident)
                ]
                mask_ious = [
                    inst_metrics.mask_iou(mask_pred, label_pred, masks_true, labels_true)
                    for mask_pred, label_pred in zip(masks_conficent, labels_confident)
                ]
            else:
                box_ious = None
                mask_ious = None
            
            # Show the predicted bounding masks and boxes with scores and box IoUs
            show_instance_masks(img, masks_conficent,
                                boxes=None if separate_boxes else boxes_confident,
                                scores=scores_confident, score_decimal=score_decimal,
                                labels=labels_confident, idx_to_class=idx_to_class_uk,
                                bg_idx=bg_idx, border_idx=border_idx,
                                mask_ious=mask_ious,
                                box_ious=box_ious if separate_boxes else None,
                                iou_decimal=iou_decimal,
                                colors=colors, fill=fill, width=width,
                                font_size=font_size, text_color=text_color, alpha=alpha,
                                ax=axes[0][1] if separate_boxes else axes[1])
            if separate_boxes:
                det.show_bounding_boxes(img, boxes_confident, labels=labels_confident,
                                        idx_to_class=idx_to_class,
                                        ious=box_ious, iou_decimal=iou_decimal,
                                        scores=scores_confident, score_decimal=score_decimal,
                                        colors=colors, fill=fill, width=width,
                                        font_size=font_size, text_color=text_color,
                                        ax=axes[1][1])
                axes[0][1].set_title(f'Predicted masks (Score > {score_thresh})')
                axes[1][1].set_title(f'Predicted boxes (Score > {score_thresh})')
            else:
                axes[1].set_title(f'Predicted masks and boxes (Score > {score_thresh})')
            
            fig.suptitle(f'Comparison of Ground Truth and Predicted Instances (Image {i})')
            plt.show()
            figures.append(fig)
        
        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

    return figures

def show_predicted_semantic_masks(imgs: torch.Tensor,
                                  preds: List[Dict[Literal['masks', 'labels', 'scores'], torch.Tensor]],
                                  targets: List[Dict[Literal['masks', 'labels'], torch.Tensor]],
                                  idx_to_class: Dict[int, str],
                                  bg_idx:int = 0,
                                  border_idx:int = None,
                                  score_threshold:float = 0.2,
                                  occlusion_priority=None) -> List[Figure]:
    """
    Show predicted instance masks as semantic segmentation masks.

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

    occlusion_priority : List[int]
        The list of label indices that indicate the priority when the occlusion occurs.
        The first label in the list has the highest priority.
        If None, the priority is based on the order of the labels.

    Returns
    -------
    figures : List[Figure]
        List of the figures that show the predicted instance masks as semantic segmentation masks.
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
            pred['masks'].cpu(), pred['labels'].cpu(), bg_idx, 
            border_idx=border_idx, border_mask=None,
            add_occlusion=False, occlusion_priority=occlusion_priority)
        for pred in preds_confident
    ]
    target_semantic_masks = [
        convert_instance_masks_to_semantic_mask(
            target['masks'].cpu(), target['labels'].cpu(), bg_idx, 
            border_idx=border_idx, border_mask=target['border_mask'] if 'border_mask' in target else None,
            add_occlusion=False, occlusion_priority=occlusion_priority)
        for target in targets
    ]
    # Show the predicted and target semantic segmentation masks
    figures = show_predicted_segmentations(imgs, pred_semantic_masks, target_semantic_masks,
                                           idx_to_class, pred_type='label', bg_idx=bg_idx, border_idx=border_idx)
    return figures
