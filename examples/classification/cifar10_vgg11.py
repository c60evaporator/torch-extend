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
EPOCHS = 30
BATCH_SIZE = 128  # Bigger batch size is faster but less accurate (https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU)
NUM_WORKERS = 4
DATA_ROOT = './datasets/CIFAR10'
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
MODEL_NAME = 'fasterrcnn_resnet50_fpn'
DROPOUT = 0.5

# Log Parameters
MLFLOW_TRACKING_URI = './log/mlruns'
MLFLOW_EXPERIMENT_NAME = 'cifar10'
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
    mlflow.log_param('DROPOUT', DROPOUT)
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
import cv2

from torch_extend.validate.common import validate_same_img_size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Transforms for training
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # ImageNet Normalization
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
])
# Transforms for validation and test (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ToTensorV2()
])

# Log the transforms
if MLFLOW_TRACKING_URI is not None:
    with mlflow.start_run(run_id=run.info.run_id):
        mlflow.log_param('train_transform', train_transform)
        mlflow.log_param('eval_transform', eval_transform)

# Validate the same image size
same_img_size_train = validate_same_img_size(train_transform)
same_img_size_eval = validate_same_img_size(eval_transform)
if not same_img_size_train:
    raise ValueError("Resize to the same size is necessary in train_transform")

# %% Define the dataset
###### 3. Define the dataset ######
from torch_extend.dataset import CIFAR10TV
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Dataset
train_dataset = CIFAR10TV(
    DATA_ROOT,
    train=True,
    transform=train_transform,
    target_transform=None,
    download=True
)
val_dataset = CIFAR10TV(
    DATA_ROOT,
    train=False,
    transform=eval_transform,
    target_transform=None,
    download=True
)
# Class to index dict
class_to_idx = train_dataset.class_to_idx
num_classes = max(class_to_idx.values()) + 1
# Index to class dict
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

# Dataloader
def collate_fn(batch):
    return tuple(zip(*batch))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=None if same_img_size_eval else collate_fn)

# Denormalize the image
def denormalize_image(img, transform):
    # Denormalization based on the transforms
    for tr in transform:
        if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
            reverse_transform = v2.Compose([
                v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                    std=[1/std for std in tr.std])
            ])
            img = reverse_transform(img)
    return img

# Display the first minibatch
def show_image_and_target(img, target, ax=None):
    """Function for showing the image and target"""
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Denormalize the image
    img = denormalize_image(img, train_transform)
    # Show the image
    img_permute = img.permute(1, 2, 0)
    ax.imshow(img_permute)
    ax.set_title(f'label: {idx_to_class[target.item()]}')

train_iter = iter(train_dataloader)
imgs, targets = next(train_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    show_image_and_target(img, target)
    plt.show()

# %% Define the model
###### 4. Define the model ######
from torchvision import models
from torch import nn

if MODEL_NAME == 'vgg11':
    model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg11_bn':
    model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg13':
    model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg13_bn':
    model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg16':
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg16_bn':
    model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg19':
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1, dropout=DROPOUT)
elif MODEL_NAME == 'vgg19_bn':
    model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1, dropout=DROPOUT)
else:
    raise ValueError(f"Invalid model name: {MODEL_NAME}")

# Freeze the parameters
for name, param in model.named_parameters():
    param.requires_grad = False
# Replace layers for transfer learning
model.classifier = nn.Sequential(
    nn.Linear(512 * 7 * 7, 4096),
    nn.ReLU(True),
    nn.Dropout(p=DROPOUT),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(p=DROPOUT),
    nn.Linear(4096, num_classes),
)
# Initialize the weights
for m in model.classifier.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

# %% Criterion, Optimizer and lr_schedulers
###### 5. Criterion, Optimizer and lr_schedulers ######
# Criterion
criterion = nn.CrossEntropyLoss()
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2
import numpy as np
import io

def calc_train_loss(batch, model, criterion, device):
    """Calculate the training loss from the batch"""
    inputs, targets = batch[0].to(device), batch[1].to(device)
    outputs = model(inputs)
    return criterion(outputs, targets)

def training_step(batch, batch_idx, device, model, criterion):
    """Training step per batch"""
    loss = calc_train_loss(batch, model, criterion, device)
    return loss

def val_predict(batch, device, model):
    """Predict the validation batch"""
    # Predict the batch
    if isinstance(batch[0], torch.Tensor):
        preds = model(batch[0].to(device))
    elif isinstance(batch[0], tuple):
        preds = torch.stack([model(img.unsqueeze(0).to(device)).squeeze(0)
                            for img in batch[0]])
    # Get the targets
    if isinstance(batch[1], torch.Tensor):
        targets = batch[1].to(device)
    elif isinstance(batch[1], tuple):
        targets = torch.tensor(batch[1], dtype=torch.long).to(device)
    return preds, targets
    
def calc_val_loss(preds, targets, criterion):
    """Calculate the validation loss from the batch"""
    return criterion(preds, targets)

def convert_preds_targets_to_torchvision(preds, targets):
    """Convert the predictions and targets to TorchVision format"""
    return preds, targets

def convert_images_to_torchvision(batch):
    """Convert the images to TorchVision format"""
    return batch[0]

def get_preds_cpu(preds):
    """Get the predictions and store them to CPU as a list"""
    return [pred.cpu() for pred in preds]

def get_targets_cpu(targets):
    """Get the targets and store them to CPU as a list"""
    return [target.item() for target in targets]

def plot_predictions(images, preds, targets, n_images=4):
    """Plot the images with predictions and ground truths"""
    figures = []
    for i, (img, pred, target) in enumerate(zip(images, preds, targets)):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        predicted_label = torch.argmax(pred).item()
        img_permute = img.permute(1, 2, 0)
        ax.imshow(img_permute)
        ax.set_title(f'pred: {idx_to_class[predicted_label]}, true: {idx_to_class[target.item()]}')
        plt.show()
        figures.append(fig)
        if i >= n_images:
            break
    return figures

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
    return loss

def calc_epoch_metrics(preds, targets):
    """Calculate the metrics from the targets and predictions"""
    # Calculate the accuracy, precision, recall, and f1 score
    predicted_labels = [torch.argmax(pred).item() for pred in preds]
    accuracy = accuracy_score(targets, predicted_labels)
    precision_macro = precision_score(targets, predicted_labels, average='macro')
    recall_macro = recall_score(targets, predicted_labels, average='macro')
    f1_macro = f1_score(targets, predicted_labels, average='macro')
    global cm
    cm = confusion_matrix(targets, predicted_labels)
    return {'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro}

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

#%% Plot predicted labels in the first minibatch of the validation dataset
model.eval()  # Set the evaluation mode
val_iter = iter(val_dataloader)
imgs, targets = next(val_iter)
preds, targets = val_predict((imgs, targets), device, model)
imgs = [denormalize_image(img, eval_transform) for img in imgs]
plot_predictions(imgs, preds, targets)

# %% Plot the confusion matrix
# Plot the confusion matrix
import seaborn as sns
import pandas as pd

# Create a DataFrame from the confusion matrix
df_cm = pd.DataFrame(cm, index=[idx_to_class[i] for i in range(len(idx_to_class))], 
                     columns=[idx_to_class[i] for i in range(len(idx_to_class))])
# Plot the confusion matrix
plt.figure(figsize=(len(idx_to_class), len(idx_to_class)*0.8))
sns.heatmap(df_cm, annot=True, fmt=".5g", cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# Calculate and print the accuracy of each class
print('Accuracy of each class')
class_accuracies = df_cm.values.diagonal() / df_cm.sum(axis=1).values
class_accuracies = sorted(class_accuracies)
print(pd.Series(class_accuracies, index=idx_to_class.values()))

# %%
