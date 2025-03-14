#%% Select the device and hyperparameters
###### 1. Select the device and hyperparameters ######
import os
import sys
import torch

# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

# General Parameters
EPOCHS = 40
BATCH_SIZE = 128  # Bigger batch size is faster but less accurate (https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU)
NUM_WORKERS = 4
DATA_ROOT = './datasets/CIFAR10'
# Optimizer Parameters
OPT_NAME = 'sgd'
LR = 0.03
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
DROPOUT = 0.5

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

NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]
# Transforms for training (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
train_transform = A.Compose([
    A.Resize(32,32),
    A.HorizontalFlip(),
    A.Rotate(limit=5, interpolation=cv2.INTER_NEAREST),
    A.Affine(rotate=0, shear=10, scale=(0.9,1.1)),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization from uint8 [0, 255] to float32 [-1.0, 1.0]
    ToTensorV2()  # Convert from numpy.ndarray to torch.Tensor
])
# Transforms for validation and test (https://www.kaggle.com/code/zlanan/cifar10-high-accuracy-model-build-on-pytorch)
eval_transform = A.Compose([
    A.Resize(32,32),
    A.Normalize(NORM_MEAN, NORM_STD),
    ToTensorV2()
])

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
                            shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=None if same_img_size_eval else collate_fn)

# Display the first minibatch
def show_image_and_target(img, target, ax=None):
    """Function for showing the image and target"""
    # If ax is None, use matplotlib.pyplot.gca()
    if ax is None:
        ax=plt.gca()
    # Denormalize the image
    denormalize_image = v2.Compose([
        v2.Normalize(mean=[-mean/std for mean, std in zip(NORM_MEAN, NORM_STD)],
                     std=[1/std for std in NORM_STD])
    ])
    img = denormalize_image(img)
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
from torch import nn

class LeNet(nn.Module):
    """LeNet model for CIFAR-10 (https://www.kaggle.com/code/vikasbhadoria/cifar10-high-accuracy-model-build-on-pytorch)"""
    def __init__(self, dropout=0.5, num_classes=10):
        super().__init__()
        self.fetrures = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*4*64, 500),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(500, num_classes)
        )
    def forward(self, x):
      x = self.fetrures(x)
      x = x.view(-1, 4*4*64)
      x = self.classifier(x)
      return x

model = LeNet(dropout=DROPOUT, num_classes=num_classes)

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

def get_preds_cpu(preds):
    """Get the predictions and store them to CPU as a list"""
    return [pred.cpu() for pred in preds]

def get_targets_cpu(targets):
    """Get the targets and store them to CPU as a list"""
    return [target.item() for target in targets]

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

#%% Plot predicted labels in the first minibatch of the validation dataset
model.eval()  # Set the evaluation mode
val_iter = iter(val_dataloader)
imgs, targets = next(val_iter)
preds, targets = val_predict((imgs, targets), device, model)
for i, (img, pred, target) in enumerate(zip(imgs, preds, targets)):
    predicted_label = torch.argmax(pred).item()
    # Denormalize the image
    denormalize_image = v2.Compose([
        v2.Normalize(mean=[-mean/std for mean, std in zip(NORM_MEAN, NORM_STD)],
                     std=[1/std for std in NORM_STD])
    ])
    img_permute = img.permute(1, 2, 0)
    plt.imshow(img_permute)
    plt.title(f'pred: {idx_to_class[predicted_label]}, true: {idx_to_class[target.item()]}')
    plt.show()

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
