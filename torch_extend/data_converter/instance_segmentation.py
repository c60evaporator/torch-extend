from typing import Literal
import torch
from torchvision.ops import box_convert

def convert_instance_masks_to_semantic_mask(masks, labels, 
                                            bg_idx=0, border_idx=255, border_mask=None,
                                            add_occlusion=True, occlusion_priority=None):
    """
    Convert instance masks to a single semantic segmentation mask.

    Parameters
    ----------
    masks : torch.Tensor (N, H, W)
        Target masks of the objects
    labels : torch.Tensor (N)
        Target labels of the objects
    bg_idx : int
        The index of the background in the target mask.
    border_idx : int
        The index of the border in the target mask.
    border_mask : torch.Tensor (H, W)
        Boolean (uint8 0/1) mask for the border area
    add_occlusion : bool
        If True, the occlusion index (254) is added to the output mask
    occlusion_priority : List[int]
        The list of label indices that indicate the priority when the occlusion occurs.
        The first label in the list has the highest priority.
        If None, the priority is based on the order of the labels.
    """
    if bg_idx is None:
        bg_idx = -1
    # Send the masks and labels to the CPU
    masks = masks.cpu().detach()
    labels = labels.cpu().detach()
    # If occlusion_priority is None, set the priority based on the order of the labels
    if occlusion_priority is None:
        occlusion_priority = torch.unique(labels).numpy().tolist()
    # Sort the masks and labels based on the occlusion priority (Reverse order)
    indices_priority = [occlusion_priority.index(label.item()) for label in labels]
    sorted_labels = [label for _, label in sorted(zip(indices_priority, labels), key=lambda x: x[0], reverse=True)]
    sorted_masks = [mask for _, mask in sorted(zip(indices_priority, masks), key=lambda x: x[0], reverse=True)]
    # Accumulate the masks
    seg_mask = torch.full(masks.size()[-2:], bg_idx, dtype=torch.int64)
    accumulated_mask = torch.full(masks.size()[-2:], 0, dtype=torch.int64)
    for mask, label in zip(sorted_masks, sorted_labels):
        seg_mask.masked_fill_(mask.bool(), label)
        accumulated_mask += mask
    # Add occlusion mask
    occulded_mask = accumulated_mask > 1
    if add_occlusion:
        seg_mask.masked_fill_(occulded_mask, 254)
    # border
    if border_mask is not None:
        seg_mask.masked_fill_(border_mask.bool(), border_idx)
    return seg_mask

def convert_image_target_to_transformers(image, target, processor, same_img_size, out_fmt='maskformer'):
    """
    Convert image and target from TorchVision to Transformers format (Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)

    Parameters
    ----------
    image : Dict
        Source image data of numpy.ndarray or torch.Tensor with TorchVision format (C, H, W)

    target : Dict
        Source target data with TorchVision format

        {'boxes': torch.Tensor([[xmin1, ymin1, xmax1, ymax1],..]), 'labels': torch.Tensor([labelindex1, labelindex2,..])}

    processor : BaseImageProcessor
        The processor for the Transformers instance segmentation model

    same_img_size : bool
        Whether all the images have the same size

    out_fmt : Literal['maskformer']
        Format of the output data.
        
        'maskformer' is the format for the `MaskFormerForInstanceSegmentation` and `Mask2FormerForUniversalSegmentation` model.
    
    Returns
    -------
    item: Dict
        Output data in the Transformers instance segmentation format
        
        maskformer: {"pixel_values": torch.Tensor(C, H, W), "pixel_mask": torch.Tensor(H, W), "mask_labels": torch.Tensor(n_instances, H, W), "class_labels": torch.Tensor(n_instances)}
    """
    if out_fmt == 'maskformer':
        # Convert instance masks into one mask with instance IDs
        segmentation_map = torch.zeros_like(target['masks'][0])
        for i, mask in enumerate(target['masks']):
            segmentation_map[mask.bool()] = i+1
        # mapping between instance IDs and class labels
        inst2class = {i: label for i, label in enumerate([0] + target['labels'].tolist())}
        # Apply the Transformers processor
        encoding = processor(images=image, segmentation_maps=segmentation_map,
                             instance_id_to_semantic_id=inst2class,
                             return_tensors="pt")
        # If images sizes are the same
        if same_img_size:
            # Remove batch dimension and return as dictionary
            return {
                "pixel_values": encoding["pixel_values"].squeeze(),
                "pixel_mask": encoding["pixel_mask"].squeeze(),
                "mask_labels": encoding["mask_labels"][0],
                "class_labels": encoding["class_labels"][0]
            }
        # If images sizes are different, `processor.encode_inputs(pixel_values, segmentation_map, return_tensors="pt")` should be applied in `collate_fn` of the DataLoader.
        else:
            # Recreate the segmentation map
            segmentation_map = torch.zeros_like(encoding["pixel_mask"].squeeze())
            for i, mask in enumerate(encoding["mask_labels"][0][1:, :, :]):
                segmentation_map[mask.bool()] = i+1
            # Remove batch dimension and return as dictionary
            return {
                "images": encoding["pixel_values"].squeeze(),
                "segmentation_maps": segmentation_map,
                "instance_id_to_semantic_id": inst2class
            }

def convert_batch_to_torchvision(batch, in_fmt='transformers'):
    """
    Convert the batch to the torchvision format images and targets

    Parameters
    ----------
    batch : Dict
        Source batch data (transformers object detection format)

    in_fmt : Literal['transformers']
        Format of the input batch data.

        'transformers': {"pixel_values": torch.Tensor(B, C, H, W), "pixel_mask": torch.Tensor(B, H, W), "mask_labels": torch.Tensor(B, n_instances, H, W), "class_labels": torch.Tensor(n_instances)},...]} with boxes in normalized cxcywh format
    
    Returns
    -------
    images: List[torch.Tensor(C, H, W)]
        Images in the batch
    
    targets: List[{{"boxes": torch.Tensor(n_instances, 4), "labels": torch.Tensor(n_instances)}]
        Targets in the batch

        The box format is [xmin, ymin, xmax, ymax]
    """
    images = []
    targets = []
    if in_fmt == 'transformers':
        device = batch['pixel_values'].device
        for pixel_value, pixel_mask, mask_label, label in zip(batch['pixel_values'], batch['pixel_mask'], batch['mask_labels'], batch['class_labels']):
            # Get the pixel mask rectangle
            nonzero_rows = torch.nonzero(pixel_mask.sum(dim=1))
            y_min, y_max = nonzero_rows[0].item(), nonzero_rows[-1].item()
            nonzero_cols = torch.nonzero(pixel_mask.sum(dim=0))
            x_min, x_max = nonzero_cols[0].item(), nonzero_cols[-1].item()
            # Extract the effective area
            image = pixel_value[:, y_min:y_max+1, x_min:x_max+1]
            # Convert the mask_labels based on the effective area
            target = {}
            target['masks'] = mask_label[:, y_min:y_max+1, x_min:x_max+1]
            # Mask float32(N, 1, H, W) -> uint8(N, H, W)
            target['masks'] = torch.round(target['masks']).to(torch.uint8)
            # Create boxes from the masks
            nonzero_masks = [mask.nonzero() for mask in target['masks']]
            target['boxes'] = torch.tensor([
                [nonzero[:, 1].min(), nonzero[:, 0].min(),
                nonzero[:, 1].max(), nonzero[:, 0].max()]
                for nonzero in nonzero_masks
            ], dtype=torch.float32, device=device)
            # Labels
            target['labels'] = label

            images.append(image)
            targets.append(target)
    
    return images, targets
