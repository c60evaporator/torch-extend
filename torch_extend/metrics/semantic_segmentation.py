from typing import Dict, List, Union, Literal
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import time
import tqdm

def segmentation_ious_one_image_or_batch(labels_pred: Tensor, target: Tensor, label_indices: List[int], border_idx:int = None):
    """
    Calculate segmentation IoUs, TP, FP, FN in one image or batch

    Reference: https://stackoverflow.com/questions/31653576/how-to-calculate-the-mean-iu-score-in-image-segmentation
    
    Parameters
    ----------
    labels_pred : Tensor(H, W) or Tensor(B, H, W)
        The predicted labels of each pixel

    target : Tensor(H, W) or Tensor(B, H, W)
        The true labels of each pixel
    
    label_indices : List[int]
        The list of label indices

    border_idx : int
        Index of the border class. The border area in the target image is ignored in the calculation or IoU.
    """
    labels_flatten = labels_pred.cpu().detach().numpy().flatten()
    target_flatten = target.cpu().detach().numpy().flatten()
    # The border area in the target image is ignored in the calculation or IoU
    if border_idx is not None:
        labels_flatten = labels_flatten[target_flatten != border_idx]
        target_flatten = target_flatten[target_flatten != border_idx]
    confmat = confusion_matrix(target_flatten, labels_flatten, labels=label_indices)
    tps = np.diag(confmat)
    gts = np.sum(confmat, axis=1)  # Number of pixels of each class in the target
    preds = np.sum(confmat, axis=0)
    unions = gts + preds - tps
    fps = preds - tps
    fns = gts - tps
    ious = np.divide(tps, unions.astype(np.float32), out=np.full((len(label_indices),), np.nan), where=(unions!=0))
    return ious, tps, fps, fns, confmat

def segmentation_ious(preds: List[Tensor],
                      targets: List[Tensor],
                      idx_to_class: Dict[int, str],
                      pred_type: Literal['label', 'logit'] = 'logit',
                      bg_idx:int = 0,
                      border_idx:int = None):
    """
    Calculate mean IoUs of each class label

    Parameters
    ----------
    preds : List[Tensor(class, H, W)] or List[Tensor(B, class, H, W)]
        List of the predicted segmentation images.

        If `pred_type` is 'logit', preds are the logit values of List[Tensor(class, H, W)] or Tensor(B, class, H, W).

        If `pred_type` is 'label', preds are the predicted labels of List[Tensor(H, W)] or Tensor(B, H, W).

    targets : List[Tensor(H, W)] or List[Tensor(B, class, H, W)]
        List of the true segmentation images

    idx_to_class : Dict[int, str]
        A dict for converting class IDs to class names.

    pred_type : Literal['label', 'logit']
        The type of the prediction. If 'label', preds are the predicted labels. If 'logit', preds are the logit values.

    bg_idx : int
        The index of the background in the target mask.

    border_idx : int
        Index of the border class. The border area in the target image is ignored in the calculation or IoU.

    Returns
    -------
    tps_batch : np.ndarray
        Number of true positives of each class in the batch
    fps_batch : np.ndarray
        Number of false positives of each class in the batch
    fns_batch : np.ndarray
        Number of false negatives of each class in the batch
    ious_batch : np.ndarray
        IoUs of each class in the batch
    label_indices : List[int]
        The list of label indices
    confmat_batch : np.ndarray
        Confusion matrix of all the pixels in the batch
    """
    if not isinstance(preds, list) or len(preds)==0 or not isinstance(preds[0], Tensor):
        raise ValueError('The `preds` must be lists of prediction mask of Tensor(H, W) or mask batch of Tensor(B, H, W).')
    if not isinstance(targets, list) or len(targets)==0 or not isinstance(targets[0], Tensor):
        raise ValueError('The `targets` must be lists of target mask of Tensor(H, W) or mask batch of Tensor(B, H, W).')
    
    if len(preds) != len(targets):
        raise ValueError('The number of preds and targets must be the same.')
    
    if preds[0].dim() == 2:
        tensor_unit = 'image'
        if targets[0].dim() != 2:
            raise ValueError('The dimension of the target tensor must be 2 (H, W) if the prediction tensor is 2 (H, W).')
    elif preds[0].dim() == 3:
        tensor_unit = 'batch'
        if targets[0].dim() != 3:
            raise ValueError('The dimension of the target tensor must be 3 (B, H, W) if the prediction tensor is 3 (B, H, W).')
    else:
        raise ValueError('The dimension of the prediction tensor must be 2 (H, W) or 3 (B, H, W).')

    # Add the background to the labels if exists
    label_indices = list(idx_to_class.keys())
    if bg_idx is not None and bg_idx not in label_indices:
        label_indices.insert(0, bg_idx)

    # calculate IoUs per image or batch
    tps_batch = []
    fps_batch = []
    fns_batch = []
    confmat_batch = np.zeros((len(label_indices), len(label_indices)))
    # Loop of images
    for i, (pred, target) in enumerate(zip(preds, targets)):
        # Get the predicted labels
        if pred_type == 'label':
            labels_pred = pred
        elif pred_type == 'logit':
            labels_pred = pred.argmax(0) if tensor_unit == 'image' else pred.argmax(1)
        else:
            raise ValueError('The `pred_type` must be either "label" or "logit".')
        # Calculate the IoUs
        ious, tps, fps, fns, confmat = segmentation_ious_one_image_or_batch(
            labels_pred, target, label_indices=label_indices, border_idx=border_idx)
        tps_batch.append(tps)
        fps_batch.append(fps)
        fns_batch.append(fns)
        confmat_batch += confmat
        if i > 0 and i%100 == 0:  # Show progress every 100 images
            print(f'Calculating IOUs: {i}/{len(preds)}{tensor_unit}s')
    # Accumulate IoUs
    tps_batch = np.array(tps_batch).sum(axis=0)
    fps_batch = np.array(fps_batch).sum(axis=0)
    fns_batch = np.array(fns_batch).sum(axis=0)
    union_batch = tps_batch + fps_batch + fns_batch
    ious_batch = np.divide(tps_batch, union_batch.astype(np.float32), out=np.full((len(tps_batch),), np.nan), where=(union_batch!=0))
    
    return tps_batch, fps_batch, fns_batch, ious_batch, label_indices, confmat_batch

def segmentation_ious_torchvison(dataloader: DataLoader, model: nn.Module, device: Literal['cuda', 'cpu'],
                                 idx_to_class: Dict[int, str], border_idx:int = None):
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

    border_idx : int
        Index of the border class. The border area in the target image is ignored in the calculation or IoU.

    Returns
    -------
    aps : Dict[int, Dict[Literal['label_name', 'average_precision', 'precision', 'recall'], Any]]
        Calculated average precisions with the label_names and the PR Curve
    """
    # Predict
    tps_all = []
    fps_all = []
    fns_all = []
    start = time.time()  # For elapsed time
    # Batch iteration
    torch.set_grad_enabled(False)
    model.eval()
    for i, (imgs, targets) in enumerate(dataloader):
        # Predict
        if isinstance(imgs, tuple) or isinstance(imgs, list):  # Separated batch with collate_fn
            preds = [model(img.to(device).unsqueeze(0)) for img in imgs]
            preds = [pred['out'] if isinstance(pred, dict) else pred for pred in preds]
        else:  # Single batch
            preds = model(imgs.to(device))
            if isinstance(preds, torch.Tensor):
                pass
            elif isinstance(preds, dict) and 'out' in preds.keys():
                preds = preds['out']
            else:
                raise ValueError('The model output is neither a Tensor nor a dict with "out" key. Please check the model output format.')
        # Calculate TP, FP, FN of the batch
        tps_batch, fps_batch, fns_batch, ious_batch, label_indices, confmat = segmentation_ious(
            preds, targets, idx_to_class, bg_idx=0, border_idx=border_idx)
        tps_all.append(tps_batch)
        fps_all.append(fps_batch)
        fns_all.append(fns_batch)
        if i%100 == 0:  # Show progress every 100 images
            print(f'Prediction for mIoU: {i}/{len(dataloader)} batches, elapsed_time: {time.time() - start}')
    tps_all = np.array(tps_all).sum(axis=0)
    fps_all = np.array(fps_all).sum(axis=0)
    fns_all = np.array(fns_all).sum(axis=0)
    union_all = tps_all + fps_all + fns_all
    ious_all = np.divide(tps_all, union_all.astype(np.float32), out=np.full((len(tps_all),), np.nan), where=(union_all!=0))

    # Store the result
    iou_dict = {
        'per_class': {
            label: {
                'label_index': label,
                'label_name': idx_to_class[label] if label in idx_to_class.keys() else 'background' if label == 0 else 'unknown',
                'tps': tps_all[i],
                'fps': fps_all[i],
                'fns': fns_all[i],
                'iou': ious_all[i]
            }
            for i, label in enumerate(label_indices) 
        },
        'confmat': confmat
    }

    torch.set_grad_enabled(True)
    model.train()

    return iou_dict
