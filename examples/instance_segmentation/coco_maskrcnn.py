#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch
import mlflow
from datetime import datetime

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 1
BATCH_SIZE = 4  # Bigger batch size increase the training time in Object Detection. Very mall batch size (E.g., n=1, 2) results in bad accuracy and poor Batch Normalization.
NUM_WORKERS = 2  # 2 * Number of devices (GPUs) is appropriate in general, but this number doesn't matter in Object Detection.
DATA_ROOT = '../detection/datasets/COCO'
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9  # For SGD and RMSprop
RMSPROP_ALPHA = 0.99  # For RMSprop
EPS = 1e-8  # For RMSprop, Adam, and AdamW
ADAM_BETAS = (0.9, 0.999)  # For Adam and AdamW
# LR Scheduler Parameters
LR_SCHEDULER = None
LR_GAMMA = 0.1
LR_STEP_SIZE = 8  # For StepLR
LR_STEPS = [16, 24]  # For MultiStepLR
LR_T_MAX = EPOCHS  # For CosineAnnealingLR
LR_PATIENCE = 10  # For ReduceLROnPlateau
# Model Parameters
MODEL_NAME = 'maskrcnn_resnet50_fpn'
SEMANTIC_METRICS_SCORE_THRESHOLD = 0.2

# Log Parameters
MLFLOW_TRACKING_URI = './log/mlruns'
MLFLOW_EXPERIMENT_NAME = 'voc_instance'
MLFLOW_ARTIFACT_LOCATION = None

# Start MLFlow experiment run
if MLFLOW_TRACKING_URI is not None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:  # Create a new experiment if it doesn't exist
        experiment_id = mlflow.create_experiment(
                                name=MLFLOW_EXPERIMENT_NAME,
                                artifact_location=MLFLOW_ARTIFACT_LOCATION)
    else: # Get an experiment ID if it exists
        experiment_id = experiment.experiment_id
    # Start the run
    run = mlflow.start_run(experiment_id=experiment_id, run_name=f'{MODEL_NAME}_{datetime.now().strftime("%Y%m%d%H%M")}')
    # Log the hyperparameters
    mlflow.log_param('model_name', MODEL_NAME)
    mlflow.log_param('EPOCHS', EPOCHS)
    mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
    mlflow.log_param('NUM_WORKERS', NUM_WORKERS)
    mlflow.log_param('OPT_NAME', OPT_NAME)
    mlflow.log_param('LR', LR)
    mlflow.log_param('WEIGHT_DECAY', WEIGHT_DECAY)
    mlflow.log_param('MOMENTUM', MOMENTUM)
    mlflow.log_param('RMSPROP_ALPHA', RMSPROP_ALPHA)
    mlflow.log_param('EPS', EPS)
    mlflow.log_param('ADAM_BETAS', ADAM_BETAS)
    mlflow.log_param('LR_SCHEDULER', LR_SCHEDULER)
    mlflow.log_param('LR_GAMMA', LR_GAMMA)
    mlflow.log_param('LR_STEP_SIZE', LR_STEP_SIZE)
    mlflow.log_param('LR_STEPS', LR_STEPS)
    mlflow.log_param('LR_T_MAX', LR_T_MAX)
    mlflow.log_param('LR_PATIENCE', LR_PATIENCE)
    mlflow.end_run()

# Select the device
DEVICE = 'cuda'
if DEVICE == 'cuda':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'mps':
    device = 'mps' if torch.backends.mps.is_available() else 'cuda'
else:
    device = 'cpu'
# Set the random seed
torch.manual_seed(42)

# %% Define the transforms
###### 2. Define the transforms ######
import albumentations as A
from albumentations.pytorch import ToTensorV2
# Note: ImageNet Normalization is not needed for TorchVision Mask R-CNN
NORM_MEAN = [0.0, 0.0, 0.0]
NORM_STD = [1.0, 1.0, 1.0]

# Transforms for training
train_transform = A.Compose([
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
    ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
# Transforms for validation and test
eval_transform = A.Compose([
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# Log the transforms
if MLFLOW_TRACKING_URI is not None:
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_param('train_transform', train_transform)
        mlflow.log_param('eval_transform', eval_transform)

# %% Define the dataset
###### 3. Define the dataset ######
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from torch_extend.dataset import CocoInstanceSegmentation
from torch_extend.display.instance_segmentation import show_instance_masks
from torch_extend.data_converter.common import denormalize_image

# Dataset
TRAIN_ANNFILE='../detection/datasets/COCO/instances_train_filtered.json'
VAL_ANNFILE='../detection/datasets/COCO/instances_val_filtered.json'
train_dataset = CocoInstanceSegmentation(
    f'{DATA_ROOT}/train2017',
    annFile=TRAIN_ANNFILE,
    transforms=train_transform,
)
val_dataset = CocoInstanceSegmentation(
    f'{DATA_ROOT}/val2017',
    annFile=VAL_ANNFILE,
    transforms=eval_transform
)
# Class to index dict
class_to_idx = train_dataset.class_to_idx
num_classes = max(class_to_idx.values()) + 1
# Index to class dict
idx_to_class = {v: k for k, v in class_to_idx.items()}
bg_idx = train_dataset.bg_idx  # Background index
border_idx = train_dataset.border_idx  # Border index

# Dataloader
def collate_fn(batch):
    return tuple(zip(*batch))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

# Display the first minibatch
def show_image_and_target(img, target, ax=None):
    """Function for showing the image and target"""
    # Denormalize the image
    img = denormalize_image(img, train_transform)
    # Show the image and target
    img = (img*255).to(torch.uint8)  # Convert from float[0, 1] to uint[0, 255]
    boxes, labels, masks = target['boxes'], target['labels'], target['masks']
    show_instance_masks(img, masks=masks, boxes=boxes,
                        border_mask=target['border_mask'] if 'border_mask' in target else None,
                        labels=labels,
                        bg_idx=bg_idx, border_idx=border_idx,
                        idx_to_class=idx_to_class, ax=ax)

train_iter = iter(train_dataloader)
imgs, targets = next(train_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    show_image_and_target(img, target)
    plt.show()

# %% Define the model
###### 4. Define the model ######
from torchvision.models.detection import mask_rcnn, faster_rcnn

if MODEL_NAME == 'maskrcnn_resnet50_fpn':
    model = mask_rcnn.maskrcnn_resnet50_fpn(weights=mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
elif MODEL_NAME == 'maskrcnn_resnet50_fpn_v2':
    model = mask_rcnn.maskrcnn_resnet50_fpn_v2(weights=mask_rcnn.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

# Freeze the parameters
for name, param in model.named_parameters():
    param.requires_grad = False
# Replace layers for transfer learning (https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#object-detection-and-instance-segmentation-model-for-pennfudan-dataset)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# %% Criterion, Optimizer and lr_schedulers
###### 5. Criterion, Optimizer and lr_schedulers ######
# Criterion (Sum of all the losses)
def criterion(outputs):
    return sum(loss for loss in outputs.values())
# Optimizer (Reference https://github.com/pytorch/vision/blob/main/references/classification/train.py)
parameters = [p for p in model.parameters() if p.requires_grad]
if OPT_NAME.startswith("sgd"):
    optimizer = torch.optim.SGD(parameters, lr=LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov="nesterov" in OPT_NAME,
    )
elif OPT_NAME == "rmsprop":
    optimizer = torch.optim.RMSprop(
        parameters, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, eps=EPS, alpha=RMSPROP_ALPHA
    )
elif OPT_NAME == "adam":
    optimizer = torch.optim.Adam(parameters, lr=LR, weight_decay=WEIGHT_DECAY, eps=EPS, betas=ADAM_BETAS)
elif OPT_NAME == "adamw":
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=WEIGHT_DECAY, eps=EPS, betas=ADAM_BETAS)
# lr_schedulers (https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling)
lr_scheduler = None
if LR_SCHEDULER == "steplr":
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
elif LR_SCHEDULER == "multisteplr":
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=LR_GAMMA)
elif LR_SCHEDULER == "exponentiallr":
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
elif LR_SCHEDULER == "cosineannealinglr":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LR_T_MAX)
elif LR_SCHEDULER == "reducelronplateau":
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=LR_GAMMA, patience=LR_PATIENCE)

# %% Training and Validation loop
###### 6. Training and Validation loop ######
import time
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
import cv2
import numpy as np
import io

from torch_extend.display.instance_segmentation import show_predicted_instances, show_predicted_semantic_masks
from torch_extend.metrics.instance_segmentation import instance_mean_ious

def calc_train_loss(batch, model, criterion, device):
    """Calculate the training loss from the batch"""
    inputs = [img.to(device) for img in batch[0]]
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items() if k in ['boxes', 'labels', 'masks']}
                for t in batch[1]]
    outputs = model(inputs, targets)
    return criterion(outputs)

def training_step(batch, batch_idx, device, model, criterion):
    """Training step per batch"""
    loss = calc_train_loss(batch, model, criterion, device)
    return loss

def val_predict(batch, device, model):
    """Predict the validation batch"""
    # Predict the batch
    return model([img.to(device) for img in batch[0]]), batch[1]

def calc_val_loss(preds, targets, criterion):
    """Calculate the validation loss from the batch"""
    return None

def convert_preds_targets_to_torchvision(preds, targets, device):
    """Convert the predictions and targets to TorchVision format"""
    # Mask float32(N, 1, H, W) -> uint8(N, H, W)
    preds = [{k: torch.round(v.squeeze(1)).to(torch.uint8)
              if k == 'masks' else v for k, v in pred.items()}
             for pred in preds]
    return preds, targets

def convert_images_for_pred_to_torchvision(batch):
    return batch[0]

def get_preds_cpu(preds):
    """Get the predictions and store them to CPU as a list"""
    return [{k: v.cpu() for k, v in pred.items()} 
            for pred in preds]
    #TODO:Memory usage exceeds 64GB for COCO val dataset

def get_targets_cpu(targets):
    """Get the targets and store them to CPU as a list"""
    return [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets]

def plot_predictions(imgs, preds, targets, n_images=4):
    figures = show_predicted_instances(imgs, preds, targets, idx_to_class,
                                       border_mask=targets['border_mask'] if 'border_mask' in targets else None,
                                       bg_idx=bg_idx, border_idx=border_idx,
                                       max_displayed_images=n_images)
    return figures

def validation_step(batch, batch_idx, device, model, criterion,
                    val_batch_preds, val_batch_targets):
    """Validation step per batch"""
    # Predict the batch
    preds, targets = val_predict(batch, device, model)
    # Calculate the loss
    loss = calc_val_loss(preds, targets, criterion)
    # Convert the predicitions and targets to TorchVision format
    preds, targets = convert_preds_targets_to_torchvision(preds, targets, device)
    # Store the predictions and targets for calculating metrics
    val_batch_preds.extend(get_preds_cpu(preds))
    val_batch_targets.extend(get_targets_cpu(targets))
    # Display the predictions of the first batch
    if batch_idx == 0:
        imgs = convert_images_for_pred_to_torchvision(batch)
        imgs = [denormalize_image(img, eval_transform) for img in imgs]
        figures = plot_predictions(imgs, preds, targets)
        # Log the images to MLFlow
        if MLFLOW_TRACKING_URI is not None:
            with mlflow.start_run(run_id=run.info.run_id):
                for i, fig in enumerate(figures):
                    img_byte_arr = io.BytesIO()
                    fig.savefig(img_byte_arr, format='png')
                    img_byte_arr = cv2.imdecode(np.frombuffer(img_byte_arr.getvalue(), np.uint8), 1)
                    img_byte_arr = img_byte_arr[:,:,::-1] # BGR->RGB
                    mlflow.log_image(img_byte_arr, key=f'img{i}', step=i_epoch)
        # Show the semantic segmentation masks
        SHOW_SEMANTIC_MASKS = True
        if SHOW_SEMANTIC_MASKS:
            show_predicted_semantic_masks(imgs, preds, targets, idx_to_class,
                                            bg_idx=bg_idx, border_idx=border_idx,
                                            score_threshold=SEMANTIC_METRICS_SCORE_THRESHOLD)
        # Close the figures to prevent memory leak
        plt.close('all')
    return loss

def calc_epoch_metrics(preds, targets):
    """Calculate the metrics from the targets and predictions"""
    # Calculate the Mean Average Precision
    map_metric = MeanAveragePrecision(iou_type=["bbox", "segm"], class_metrics=True, extended_summary=True)
    map_metric.update(preds, targets)
    map_score = map_metric.compute()
    global last_preds
    global last_targets
    last_preds = preds
    last_targets = targets
    # Calculate semantic segmentation metrics
    tps, fps, fns, mean_ious, label_indices, confmat = instance_mean_ious(
        preds, targets, idx_to_class, bg_idx=bg_idx, border_idx=border_idx,
        score_threshold=SEMANTIC_METRICS_SCORE_THRESHOLD)
    semantic_mean_iou = np.mean(mean_ious)
    global last_ious
    last_ious = {
        'per_class': {
            label: {
                'label_index': label,
                'label_name': idx_to_class[label] if label in idx_to_class.keys() else 'background' if label == bg_idx else 'unknown',
                'tps': tps[i],
                'fps': fps[i],
                'fns': fns[i],
                'iou': mean_ious[i],
            }
            for i, label in enumerate(label_indices)
        },
        'confmat': confmat
    }
    print(f'MaskAP_50-95={map_score["segm_map"].item()}, MaskAP_50={map_score["segm_map_50"].item()}, BoxAP_50-95={map_score["bbox_map"].item()}, BoxAP_50={map_score["bbox_map_50"].item()}, SemanticMeanIoU={semantic_mean_iou}')
    return {'MaskAP_50-95': map_score["segm_map"].item(), 'MaskAP_50': map_score["segm_map_50"].item(), 
            'BoxAP_50-95': map_score["bbox_map"].item(), 'BoxAP_50': map_score["bbox_map_50"].item(),
            'SemanticMeanIoU': semantic_mean_iou}

def train_one_epoch(loader, device, model,
                    criterion, optimizer, lr_scheduler):
    """Train one epoch"""
    train_step_losses = []
    torch.set_grad_enabled(True)
    with tqdm(loader, unit="batches") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            # Calculate the loss
            loss = training_step(batch, batch_idx, device, model, criterion)
            train_step_losses.append(loss.detach().item())  # Record the loss
            # clear gradients
            optimizer.zero_grad()
            # Update the weights
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())
    # lr_scheduler step
    if lr_scheduler is not None:
        lr_scheduler.step()
    return train_step_losses

def val_one_epoch(loader, device, model,
                  criterion):
    """Validate one epoch"""
    val_step_losses = []
    val_batch_preds = []
    val_batch_targets = []
    torch.set_grad_enabled(False)
    model.eval()
    with tqdm(loader, unit="batches") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            loss = validation_step(batch, batch_idx, device, model, criterion,
                                   val_batch_preds, val_batch_targets)
            if loss is not None:
                val_step_losses.append(loss.detach().item())  # Record the loss
    torch.set_grad_enabled(True)
    model.train()
    # Calculate the metrics
    val_metrics_epoch = calc_epoch_metrics(val_batch_preds, val_batch_targets)
    return val_step_losses, val_metrics_epoch

# Train and validate
model.to(device)
train_epoch_losses, val_epoch_losses, val_metrics_all = [], [], []
start = time.time()
# Epoch loop
for i_epoch in range(EPOCHS):
    # Train one epoch
    train_step_losses = train_one_epoch(train_dataloader, device, model,
                                        criterion, optimizer, lr_scheduler)
    # Calculate the average loss
    train_loss_epoch = sum(train_step_losses) / len(train_step_losses)
    train_epoch_losses.append(train_loss_epoch)
    # Validate one epoch
    val_step_losses, val_metrics_epoch = val_one_epoch(val_dataloader, device, model,
                                                       criterion)
    # Calculate the average loss
    val_loss_epoch = None
    if len(val_step_losses) > 0:
        val_loss_epoch = sum(val_step_losses) / len(val_step_losses)
        val_epoch_losses.append(val_loss_epoch)
    val_metrics_all.append(val_metrics_epoch)
    # Print the losses, elapsed time, and metrics
    elapsed_time = time.time() - start
    print(f'Epoch: {i_epoch + 1}, train_loss: {train_loss_epoch}, val_loss: {val_loss_epoch}, elapsed_time: {time.time() - start}')
    print(f'Epoch: {i_epoch + 1}, ' + ' '.join([f'{k}={v}' for k, v in val_metrics_epoch.items()]))
    # Record the loss
    if MLFLOW_TRACKING_URI is not None:
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_metric(key="train_loss", value=train_loss_epoch, step=i_epoch)
            mlflow.log_metrics(val_metrics_epoch, step=i_epoch)
            if val_loss_epoch is not None:
                mlflow.log_metric(key="val_loss", value=val_loss_epoch, step=i_epoch)

# %% Plot the training history
###### 7. Plot the training history ######
# Create a figure and axes
n_metrics = len(val_metrics_all[0])
fig, axes = plt.subplots(n_metrics+1, 1, figsize=(5, 4*(n_metrics+1)))
colors = plt.get_cmap('tab10').colors
# Plot the training and validation losses
axes[0].plot(range(1, EPOCHS+1), train_epoch_losses, label='train_loss', color=colors[0])
if len(val_epoch_losses) > 0:
    axes[0].plot(range(1, EPOCHS+1), val_epoch_losses, label='val_loss', color=colors[1])
axes[0].legend()
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Losses in each epoch')
# Plot the validation metrics
for i, metric_name in enumerate(val_metrics_all[0].keys()):
    axes[i+1].plot(range(1, EPOCHS+1),
                    [metrics[metric_name] for metrics in val_metrics_all],
                    label=f'val_{metric_name}',
                    color=colors[1])
    axes[i+1].set_xlabel('Epoch')
    axes[i+1].set_ylabel(metric_name)
    axes[i+1].set_title(f'Validation {metric_name}')
fig.tight_layout()
plt.show()

#%% Plot predicted bounding boxes in the first minibatch of the validation dataset
model.eval()  # Set the evaluation mode
val_iter = iter(val_dataloader)
imgs, targets = next(val_iter)
preds, targets = val_predict((imgs, targets), device, model)
imgs = [denormalize_image(img, eval_transform) for img in imgs]
plot_predictions(imgs, preds, targets)

#%% Plot Box Average Precisions, Semantic IOUs, and Confusion Matrix
# Plot Box Average Precisions, Semantic IOUs, and Confusion Matrix
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from torch_extend.display.detection import show_average_precisions

# Plot Box Average Precisions
show_average_precisions(last_preds, last_targets, idx_to_class)

# Display the confusion matrix
label_dict = {k: v['label_name'] for k, v in last_ious['per_class'].items()}
df_confmat = pd.DataFrame(last_ious['confmat'], index=label_dict.values(), columns=label_dict.values())
plt.figure(figsize=(len(label_dict), len(label_dict)*0.8))
sns.heatmap(df_confmat, annot=True, fmt=".2g", cmap='Blues', norm=LogNorm())
plt.xlabel('Predicted', fontsize=16)
plt.ylabel('True', fontsize=16)
plt.title('Confusion Matrix', fontsize=20)
plt.show()

# Display the Semantic IOUs
print(pd.DataFrame([v for v in last_ious['per_class'].values()]))

# %%
