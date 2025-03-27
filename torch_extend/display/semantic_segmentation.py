from typing import List, Dict, Literal
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from PIL import Image

from ..metrics.semantic_segmentation import segmentation_ious_one_image_or_batch

# def _create_segmentation_palette():
#     BLIGHTNESSES = [0, 64, 128, 192]
#     len_blt = len(BLIGHTNESSES)
#     pattern_list = []
#     for i in range(255):
#         r = BLIGHTNESSES[i % len_blt]
#         g = BLIGHTNESSES[(i // len_blt) % len_blt]
#         b = BLIGHTNESSES[(i // (len_blt ** 2)) % len_blt]
#         pattern_list.append([r, g, b])
#     pattern_list.append([255, 255, 255])
#     return np.array(pattern_list, dtype=np.uint8)

def create_segmentation_palette(colors=None):
    """
    # Color palette for segmentation masks
    """
    if colors is None:
        colors = sns.color_palette(n_colors=256).as_hex()
    palette = [list(int(ip[i:i+2],16) for i in (1, 3, 5)) for ip in colors]  # Convert hex to RGB
    return palette

def array1d_to_pil_image(array: torch.Tensor, palette: List[List[int]], 
                         bg_idx=None, border_idx=None, occlusion_idx=None,
                         bg_color=[255, 255, 255], border_color=[0, 0, 0], occlusion_color=[64, 64, 64]):
    """
    Convert 1D class image to colored PIL image

    Parameters
    ----------
    array : torch.Tensor (x, y)
        Input image whose value indicate the class label
    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for specifying the classes
    bg_idx : int
        The index of the background in the target mask.
    border_idx : int
        The index of the border in the target mask.
    occlusion_idx : int
        The index of the occlusion in the target mask (for Instance Segmentation).
    bg_color : List[int]
        The color of the background described in [R, G, B]
    border_color : List[int]
        The color of the border described in [R, G, B]
    occlusion_color : List[int]
        The color of the occlusion described in [R, G, B]
    """
    # Replace the background
    if bg_idx is not None:
        palette[bg_idx] = bg_color
    # Replace the border
    if border_idx is not None:
        palette[border_idx] = border_color
    # Replace the occlusion
    if occlusion_idx is not None:
        palette[occlusion_idx] = occlusion_color
    # Convert the array from torch.tensor to np.ndarray
    array_numpy = array.detach().to('cpu').numpy().astype(np.uint8)
    # Convert the array
    pil_out = Image.fromarray(array_numpy, mode='P')
    pil_out.putpalette(np.array(palette, dtype=np.uint8))
    return pil_out

def show_segmentation(image, target,
                      alpha=0.5, palette=None,
                      bg_idx=0, border_idx=None,
                      add_legend=True, idx_to_class=None,
                      plot_raw_image=True, iou_scores=None, score_decimal=3,
                      ax=None):
    """
    Show the image with the segmentation.

    Parameters
    ----------
    image : torch.Tensor (C, H, W)
        Input image
    target : torch.Tensor (H, W)
        Target segmentation class with Torchvision segmentation format 
    alpha : float
        Transparency of the segmentation 
    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for specifying the classes
    bg_idx : int
        Index of the background class
    border_idx : int
        Index of the border class
    add_legend : bool
        If True, the legend of the class labels is added to the plot
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        Only available if `add_legend` is true
        If None, class ID is used for the plot
    plot_raw_image : bool
        If True, the raw image is plotted as the background
    iou_scores : Dict[int, float]
        A dict of IoU scores whose keys are the indices of the label. If None, IoU scores are not displayed.
    score_decimal : int
        A decimal for the displayed confidence scores. Available only if iou_scores is True
    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.
    """
    # Auto palette generation
    if palette is None:
        palette = create_segmentation_palette()
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Display the raw image
    if plot_raw_image:
        ax.imshow(image.permute(1, 2, 0))
    # Display Segmentations
    segmentation_img = array1d_to_pil_image(target, palette, bg_idx, border_idx)
    ax.imshow(segmentation_img, alpha=alpha)
    # Add legend
    if add_legend:
        labels_unique = torch.unique(target).cpu().detach().numpy().tolist()
        # Convert class IDs to class names
        if idx_to_class is None:
            idx_to_class_bg = {idx: str(idx) 
                               for idx in range(np.max(labels_unique[labels_unique != border_idx]))}
        else:
            idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
        # Add the background and border label
        if bg_idx is not None:
            idx_to_class_bg[bg_idx] = 'background'
        # Make the label text with IoU scores
        if iou_scores is not None:
            label_dict = {k: f'{v}, IoU={round(iou_scores[k], score_decimal)}' 
                          for k, v in idx_to_class_bg.items()}
        else:
            label_dict = {k: v for k, v in idx_to_class_bg.items()}
            if border_idx is not None:
                label_dict[border_idx] = 'border'
        # Add legend
        handles = [mpatches.Patch(facecolor=(palette[label][0]/255,
                                         palette[label][1]/255,
                                         palette[label][2]/255), 
                              label=label_dict[label])
               for label in labels_unique]
        ax.legend(handles=handles)

def show_segmentations(image, target, 
                       idx_to_class=None,
                       alpha=0.5, palette=None,
                       bg_idx=0, border_idx=None,
                       plot_raw_image=True):
    """
    Show the image with the segmentation, legend, and the row image.

    Parameters
    ----------
    image : torch.Tensor (C, H, W)
        Input image with 256 color indices
    target : torch.Tensor (H, W)
        Target segmentation class with Torchvision segmentation format 
    alpha : float
        Transparency of the segmentation 
    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for specifying the classes
    bg_idx : int
        Index of the background class
    border_idx : int
        Index of the border class
    plot_raw_image : bool
        If True, the raw image is plotted as the background
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Auto palette generation
    if palette is None:
        palette = create_segmentation_palette()
    # Plot the segmentation image
    show_segmentation(image, target, alpha, palette, bg_idx, border_idx, 
                      add_legend=True, idx_to_class=idx_to_class,
                      plot_raw_image=plot_raw_image, ax=axes[0])
    axes[0].set_title('True segmentation')
    # Plot the row image
    axes[1].imshow(image.permute(1, 2, 0))
    axes[1].set_title('Raw image')
    plt.show()

def show_predicted_segmentations(imgs, preds, targets, idx_to_class,
                                 pred_type: Literal['label', 'logit'] = 'logit',
                                 alpha=0.5, palette=None,
                                 bg_idx=0, border_idx=None,
                                 plot_raw_image=True,
                                 max_displayed_images=None,
                                 calc_iou=True) -> List[Figure]:
    """
    Show predicted minibatch images with predicted and true segmentation.

    Parameters
    ----------
    imgs : torch.Tensor (B, C, H, W)
        Images which are standardized to [0, 1]
    
    preds : List[Tensor(class, H, W)], List[Tensor(B, class, H, W)], List[Tensor(H, W)], or List[Tensor(B, H, W)]
        List of the predicted segmentation results.

        If `pred_type` is 'logit', preds are the logit values of List[Tensor(class, H, W)] or List[Tensor(B, class, H, W)].

        If `pred_type` is 'label', preds are the predicted labels of List[Tensor(H, W)] or List[Tensor(B, H, W)].
    
    targets : List[Tensor(H, W)] or List[Tensor(B, H, W)]
        Ground truths which indicates the label index of each pixel
    
    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.
        If None, class ID is used for the plot

    pred_type : Literal['label', 'logit']
        The type of the prediction. If 'label', preds are the predicted labels. If 'logit', preds are the logit values.

    calc_iou : bool
        If True, the IoU is calculated and displayed with the label

    alpha : float
        Transparency of the segmentation 

    palette : List ([[R1, G1, B1], [R2, G2, B2],..])
        Color palette for specifying the classes

    bg_idx : int
        Index of the background class
    
    border_idx : int
        Index of the border class

    plot_raw_image : bool
        If True, the raw image is plotted as the background

    max_displayed_images : int
        number of maximum displayed images. This is in case of big batch size.

    calc_iou : bool
        If True, the IoU is calculated and displayed with the label
    """
    # Auto palette generation
    if palette is None:
        palette = create_segmentation_palette()

    figures = []
    # Image loop
    for i, (img, pred, target) in enumerate(zip(imgs, preds, targets)):
        img = (img*255).to(torch.uint8).cpu().detach()  # Change from float[0, 1] to uint[0, 255]
        predicted_labels = pred.argmax(0).cpu().detach() if pred_type == 'logit' else pred.cpu().detach()
        # Create a camvas
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Plot the true segmentation
        show_segmentation(img, target, alpha, palette, bg_idx, border_idx, 
                          add_legend=True, idx_to_class=idx_to_class, 
                          plot_raw_image=plot_raw_image, ax=axes[0])
        axes[0].set_title('True segmentation')
        # Plot the row image
        axes[1].imshow(img.permute(1, 2, 0))
        axes[1].set_title('Raw image')
        # Calculate IoU
        if calc_iou:
            # Index to class dictionary with background and border
            idx_to_class_bd = {k: v for k, v in idx_to_class.items()}
            if bg_idx is not None:
                idx_to_class_bd[bg_idx] = 'background'
            if border_idx is not None:
                idx_to_class_bd[border_idx] = 'border'
            # Label indices for IoU calculation
            label_indices = list(idx_to_class.keys())
            if bg_idx is not None and bg_idx not in label_indices:
                label_indices.insert(0, bg_idx)
            ious, _, _, _, _ = segmentation_ious_one_image_or_batch(predicted_labels, target, label_indices, border_idx=border_idx)
            iou_scores = {
                label: ious[i]
                for i, label in enumerate(label_indices)
            }
        else:
            iou_scores = None
        # Plot the predicted segmentation
        show_segmentation(img, predicted_labels, alpha, palette, bg_idx, border_idx, 
                          add_legend=True, idx_to_class=idx_to_class, 
                          plot_raw_image=plot_raw_image, iou_scores=iou_scores, ax=axes[2])
        axes[2].set_title('Predicted segmentation')
        plt.show()
        figures.append(fig)

        if max_displayed_images is not None and i >= max_displayed_images - 1:
            break

    return figures
