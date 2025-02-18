from typing import Dict, Literal, Any
import torch
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import math
import numpy as np

from .detection import show_bounding_boxes
from .semantic_segmentation import create_segmentation_palette, array1d_to_pil_image

def show_instance_masks(image, boxes, masks, 
                        border_mask=None,
                        labels=None, idx_to_class=None,
                        colors=None, fill=False, width=1,
                        font=None, font_size=None,
                        alpha=0.4, palette=None, add_legend=True,
                        ax=None):
    """
    Show the image with the masks and bounding boxes.

    Parameters
    ----------
    image : torch.Tensor (C, H, W)
        Input image
    boxes : torch.Tensor (N, 4)
        Bounding boxes with Torchvision object detection format
    masks : torch.Tensor (N, H, W)
        Target masks of the objects
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
    show_bounding_boxes(image, boxes, labels=labels, idx_to_class=idx_to_class,
                        colors=colors, fill=fill, width=width,
                        font=font, font_size=font_size, ax=ax)
    # Auto palette generation
    if palette is None:
        palette = create_segmentation_palette()
    # Convert instance masks to single mask
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
    # Show the mask
    segmentation_img = array1d_to_pil_image(seg_mask, palette, 
                                            bg_idx=0, border_idx=255,
                                            occlusion_idx=254)
    ax.imshow(segmentation_img, alpha=alpha)
    # Add legend
    if add_legend:
        labels_unique = torch.unique(labels).cpu().detach().numpy()
        # Convert class IDs to class names
        if idx_to_class is None:
            idx_to_class_bd = {idx: str(idx) for idx in range(np.max(labels_unique))}
        else:
            idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
        # Add the border and occlusion label to idx_to_class
        idx_to_class_bd[255] = 'border'
        idx_to_class_bd[254] = 'occlusion'
        # Make the label text
        label_dict = {k: v for k, v in idx_to_class_bd.items()}
        # Add legend
        handles = [mpatches.Patch(facecolor=(palette[label][0]/255,
                                         palette[label][1]/255,
                                         palette[label][2]/255), 
                              label=label_dict[label])
               for label in labels_unique]
        ax.legend(handles=handles)
