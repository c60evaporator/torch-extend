from typing import Literal
import torch
from torchvision.ops import box_convert

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
            ], dtype=torch.float32)
            # Labels
            target['labels'] = label
            images.append(image)
            targets.append(target)
    
    return images, targets
