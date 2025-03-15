#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 4
BATCH_SIZE = 4  # Bigger batch size increase the training time in Object Detection. Very mall batch size (E.g., n=1, 2) results in bad accuracy and poor Batch Normalization.
NUM_WORKERS = 2  # 2 * Number of devices (GPUs) is appropriate in general, but this number doesn't matter in Object Detection.
DATA_ROOT = './datasets/VOC2012'
# Optimizer Parameters
OPT_NAME = 'adamw'
LR = 2e-5
WEIGHT_DECAY = 1e-4
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
LR_BACKBONE = 5e-6  # Learning rate for the backbone
# Model Parameters
MODEL_NAME = 'facebook/detr-resnet-50'

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
from transformers import AutoImageProcessor, DetrImageProcessor
import albumentations as A

# Image Processor (https://huggingface.co/docs/transformers/preprocessing#computer-vision)
image_processor = DetrImageProcessor.from_pretrained(MODEL_NAME)

# Augmentation (Resize, Normalize, and ToTensor are not needed because the image_processor does it)
# Transforms for training
train_transform = A.Compose([
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
# Transforms for validation and test
eval_transform = A.Compose([
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# %% Define the dataset
###### 3. Define the dataset ######
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from torch_extend.dataset import VOCDetection, CocoDetectionTV
from torch_extend.display.detection import show_bounding_boxes
from torch_extend.data_converter.detection import convert_batch_to_torchvision

# Dataset
train_dataset = VOCDetection(DATA_ROOT, image_set='train', download=True,
                             transforms=train_transform,
                             output_format='transformers', processor=image_processor)
val_dataset = VOCDetection(DATA_ROOT, image_set='val',
                           transforms=eval_transform,
                           output_format='transformers', processor=image_processor)
# Class to index dict
class_to_idx = train_dataset.class_to_idx
num_classes = max(class_to_idx.values()) + 1
# Index to class dict
idx_to_class = {v: k for k, v in class_to_idx.items()}



# import torchvision
# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, processor, train=True):
#         ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
#         super(CocoDetection, self).__init__(img_folder, ann_file)
#         self.processor = processor

#     def __getitem__(self, idx):
#         # read in PIL image and target in COCO format
#         # feel free to add data augmentation here before passing them to the next step
#         img, target = super(CocoDetection, self).__getitem__(idx)

#         # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
#         image_id = self.ids[idx]
#         target = {'image_id': image_id, 'annotations': target}
#         encoding = self.processor(images=img, annotations=target, return_tensors="pt")
#         pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
#         target = encoding["labels"][0] # remove batch dimension

#         return pixel_values, target

# train_dataset = CocoDetection(img_folder='datasets/balloon/train', processor=image_processor)
# val_dataset = CocoDetection(img_folder='datasets/balloon/val', processor=image_processor, train=False)
# cats = train_dataset.coco.cats
# idx_to_class = {k: v['name'] for k,v in cats.items()}



# Collate function for the DataLoader (https://huggingface.co/docs/transformers/preprocessing#computer-vision)
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)

# Denormalize the image
def denormalize_image(img, transform, processor):
    # Denormalization based on the transforms
    for tr in transform:
        if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
            reverse_transform = v2.Compose([
                v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                    std=[1/std for std in tr.std])
            ])
            img = reverse_transform(img)
    # Denormalization based on the processor or Transformers
    if processor.do_normalize:
        denormalize_image = v2.Compose([
            v2.Normalize(mean=[-mean/std for mean, std in zip(processor.image_mean, processor.image_std)],
                        std=[1/std for std in processor.image_std])
        ])
        img = denormalize_image(img)
    return img

# Display the first minibatch
def show_image_and_target(img, target, ax=None):
    """Function for showing the image and target"""
    # Show the image
    img = (img*255).to(torch.uint8)  # Convert from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels,
                        idx_to_class=idx_to_class, ax=ax)

train_iter = iter(train_dataloader)
batch = next(train_iter)
imgs, targets = convert_batch_to_torchvision(batch, in_fmt='transformers')
for i, (img, target) in enumerate(zip(imgs, targets)):
    # Denormalize the image
    img = denormalize_image(img, train_transform, image_processor)
    # Show the image and target
    show_image_and_target(img, target)
    plt.show()

# %% Define the model
###### 4. Define the model ######
from transformers import DetrConfig, DetrForObjectDetection

# DETR with randomly initialized weights for Transformer and pre-trained weights for backbone
# (https://huggingface.co/docs/transformers/model_doc/detr#usage-tips)
# config = DetrConfig(use_pretrained_backbone=True, backbone='resnet50', id2label=idx_to_class, label2id=class_to_idx)
# model = DetrForObjectDetection(config)
# ()
model = DetrForObjectDetection.from_pretrained(MODEL_NAME,
                                               revision="no_timm",
                                               num_labels=len(idx_to_class),
                                               ignore_mismatched_sizes=True)

# %% Criterion, Optimizer and lr_schedulers
###### 5. Criterion, Optimizer and lr_schedulers ######
# Criterion (Sum of all the losses)
def criterion(outputs):
    return outputs.loss
# Optimizer (Reference https://github.com/pytorch/vision/blob/main/references/classification/train.py)
parameters = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
              {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
               "lr": LR_BACKBONE}]
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
from torchvision.ops.boxes import box_convert
import torchvision.transforms.v2.functional as F
import matplotlib.pyplot as plt

from torch_extend.display.detection import show_predicted_bboxes

def calc_train_loss(batch, model, criterion, device):
    """Calculate the training loss from the batch"""
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    # for k,v in outputs.loss_dict.items():
    #       self.log("train_" + k, v.item())
    return criterion(outputs)

def training_step(batch, batch_idx, device, model, criterion):
    """Training step per batch"""
    loss = calc_train_loss(batch, model, criterion, device)
    return loss

def val_predict(batch, device, model):
    """Predict the validation batch"""
    # Predict the batch
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch["pixel_mask"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
    preds = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    return preds, labels

def calc_val_loss(preds, targets, criterion):
    """Calculate the validation loss from the batch"""
    return criterion(preds)

def convert_preds_targets_to_torchvision(preds, targets):
    """Convert the predictions and targets to TorchVision format"""
    # Post-process the predictions
    orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
    results = image_processor.post_process_object_detection(
        preds, target_sizes=orig_target_sizes, threshold=0
    )
    # Convert the targets
    targets = [{
        "boxes": box_convert(target["boxes"], 'cxcywh', 'xyxy') \
                 * torch.tensor([orig[1], orig[0], orig[1], orig[0]], dtype=torch.float32).to(DEVICE) if target["boxes"].shape[0] > 0 \
                 else torch.zeros(size=(0, 4), dtype=torch.float32).to(DEVICE),
        "labels": target["class_labels"]
    } for target, orig in zip(targets, orig_target_sizes)]
    # Return as TorchVision format
    return results, targets

def convert_images_to_torchvision(batch):
    proc_imgs, _ = convert_batch_to_torchvision(batch, in_fmt='transformers')
    orig_sizes = [label["orig_size"] for label in batch["labels"]]
    return [F.resize(img, orig_size.tolist()) for img, orig_size in zip(proc_imgs, orig_sizes)]

def get_preds_cpu(preds):
    """Get the predictions and store them to CPU as a list"""
    return [{k: v.cpu() for k, v in pred.items()} 
            for pred in preds]

def get_targets_cpu(targets):
    """Get the targets and store them to CPU as a list"""
    return [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets]

def plot_predictions(imgs, preds, targets, n_images=4):
    show_predicted_bboxes(imgs, preds, targets, idx_to_class,
                          max_displayed_images=n_images)

def validation_step(batch, batch_idx, device, model, criterion,
                    val_batch_preds, val_batch_targets):
    """Validation step per batch"""
    # Predict the batch
    preds, targets = val_predict(batch, device, model)
    # Calculate the loss
    loss = calc_val_loss(preds, targets, criterion)
    # Convert the predicitions and targets to TorchVision format
    preds, targets = convert_preds_targets_to_torchvision(preds, targets)
    # Store the predictions and targets for calculating metrics
    val_batch_preds.extend(get_preds_cpu(preds))
    val_batch_targets.extend(get_targets_cpu(targets))
    # Display the predictions of the first batch
    if batch_idx == 0:
        imgs = convert_images_to_torchvision(batch)
        imgs = [denormalize_image(img, eval_transform, image_processor) for img in imgs]
        plot_predictions(imgs, preds, targets)
    return loss

def calc_epoch_metrics(preds, targets):
    """Calculate the metrics from the targets and predictions"""
    # Calculate the Mean Average Precision
    map_metric = MeanAveragePrecision(iou_type=["bbox"], class_metrics=True, extended_summary=True)
    map_metric.update(preds, targets)
    map_score = map_metric.compute()
    global last_preds
    global last_targets
    last_preds = preds
    last_targets = targets
    print(f'BoxAP@50-95={map_score["map"].item()}, BoxAP@50={map_score["map_50"].item()}, BoxAP@75={map_score["map_75"].item()}')
    return {'BoxAP@50-95': map_score["map"].item(), 'BoxAP@50': map_score["map_50"].item(), 'BoxAP@75': map_score["map_75"].item()}

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
            # Plot the predicted bounding boxes
            # if batch_idx == 0:
            #     show_predicted_bboxes(batch[0], val_batch_preds, val_batch_targets, idx_to_class)
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

#%% Plot predicted bounding boxes in the first minibatch of the validation dataset
model.eval()  # Set the evaluation mode
val_iter = iter(val_dataloader)
batch = next(val_iter)
preds, labels = val_predict(batch, device, model)
preds, targets = convert_preds_targets_to_torchvision(preds, labels)
imgs = convert_images_to_torchvision(batch)
imgs = [denormalize_image(img, eval_transform, image_processor) for img in imgs]
show_predicted_bboxes(imgs, preds, targets, idx_to_class)

#%% Plot Average Precisions
# Plot Average Precisions
from torch_extend.display.detection import show_average_precisions

show_average_precisions(last_preds, last_targets, idx_to_class)

#%%
