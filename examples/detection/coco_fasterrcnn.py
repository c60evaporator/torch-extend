#%% Select the device
import os
import sys
# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch

# General Parameters
EPOCHS = 1
BATCH_SIZE = 4  # Bigger batch size increase the training time in Object Detection. Very mall batch size (E.g., n=1, 2) results in unstable training and bad for Batch Normalization.
NUM_WORKERS = 2  # 2 * Number of devices (GPUs) is appropriate in general, but this number doesn't matter in Object Detection.
DATA_ROOT = './datasets/COCO'
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.01
WEIGHT_DECAY = 0
MOMENTUM = 0  # For SGD and RMSprop
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
# Metrics Parameters
AP_IOU_THRESHOLD = 0.5
AP_CONF_THRESHOLD = 0.0

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

# Note: ImageNet Normalization is not needed for TorchVision Faster R-CNN
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
NORM_MEAN = [0.0, 0.0, 0.0]
NORM_STD = [1.0, 1.0, 1.0]

# Transforms for training
train_transform = A.Compose([
    A.Resize(640, 640),  # Resize the image to (640, 640)
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
    ToTensorV2(),  # Convert from numpy.ndarray to torch.Tensor
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
# Transforms for validation and test
eval_transform = A.Compose([
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization from uint8 [0, 255] to float32 [0.0, 1.0]
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))


# %% Define the Dataset
# Define the dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from torch_extend.dataset.detection.coco import CocoDetectionTV
from torch_extend.display.detection import show_bounding_boxes

# Dataset
TRAIN_ANNFILE='./datasets/COCO/instances_train_filtered.json'
VAL_ANNFILE='./datasets/COCO/instances_val_filtered.json'
train_dataset = CocoDetectionTV(
    f'{DATA_ROOT}/train2017',
    annFile=TRAIN_ANNFILE,
    transforms=train_transform
)
val_dataset = CocoDetectionTV(
    f'{DATA_ROOT}/val2017',
    annFile=VAL_ANNFILE,
    transforms=eval_transform
)
# Class to index dict
class_to_idx = train_dataset.class_to_idx
# Index to class dict
idx_to_class = {v: k for k, v in class_to_idx.items()}
na_cnt = 0
for i in range(max(class_to_idx.values())):
    if i not in class_to_idx.values():
        na_cnt += 1
        idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
# Index to class dict with background
idx_to_class_bg = {k: v for k, v in idx_to_class.items()}
idx_to_class_bg[-1] = 'background'

# Dataloader
def collate_fn(batch):
    return tuple(zip(*batch))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

# Display the first minibatch
def show_image_and_target(img, target, ax=None):
    """Function for showing the image and target"""
    # Show the image
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels,
                        idx_to_class=idx_to_class, ax=ax)

train_iter = iter(train_dataloader)
imgs, targets = next(train_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    show_image_and_target(img, target)
    plt.show()

# %% Define the model
# Define the model
from torchvision.models.detection import faster_rcnn

model = faster_rcnn.fasterrcnn_resnet50_fpn(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
# Freeze the parameters
for name, param in model.named_parameters():
    param.requires_grad = False
# Replace layers for transfer learning
num_classes = max(class_to_idx.values()) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# %% Criterion, Optimizer and lr_schedulers
###### 5. Criterion, Optimizer and lr_schedulers ######
# Criterion (Sum of all the losses)
def criterion(loss_dict):
    return sum(loss for loss in loss_dict.values())
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
import numpy as np

from torch_extend.metrics.detection import average_precisions

def calc_train_loss(batch, model, criterion, device):
    """Calculate the training loss from the batch"""
    inputs = [img.to(device) for img in batch[0]]
    targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                for t in batch[1]]
    loss_dict = model(inputs, targets)
    return criterion(loss_dict)

def training_step(batch, batch_idx, device, model, criterion):
    """Training step per batch"""
    loss = calc_train_loss(batch, model, criterion, device)
    return loss

def get_preds_cpu(inputs, model):
    """Get the predictions and store them to CPU as a list"""
    return [{k: v.cpu() for k, v in pred.items()} 
            for pred in model(inputs)]

def get_targets_cpu(targets):
    """Get the targets and store them to CPU as a list"""
    return [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets]

def validation_step(batch, batch_idx, device, model, criterion,
                    val_batch_preds, val_batch_targets):
    """Validation step per batch"""
    # Calculate the loss
    loss = None
    # Store the predictions and targets for calculating metrics
    val_batch_preds.extend(get_preds_cpu([img.to(device) for img in batch[0]], model))
    val_batch_targets.extend(get_targets_cpu(batch[1]))
    return loss

def calc_epoch_metrics(preds, targets):
    """Calculate the metrics from the targets and predictions"""
    # Calculate the mean Average Precision
    aps = average_precisions(preds, targets,
                             idx_to_class_bg, 
                             iou_threshold=AP_IOU_THRESHOLD, conf_threshold=AP_CONF_THRESHOLD)
    mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
    global last_aps
    last_aps = aps
    print(f'mAP={mean_average_precision}')
    return {'mAP': mean_average_precision}

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
            #tepoch.set_postfix(loss=loss.item())
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

#%% Plot Average Precisions
# Plot Average Precisions
from torch_extend.display.detection import show_average_precisions

show_average_precisions(last_aps)
