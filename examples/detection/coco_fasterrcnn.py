#%% Select the device
import os
import sys
# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch

# Parameters
BATCH_SIZE = 8
NUM_WORKERS = 4
DATA_ROOT = 'datasets/COCO'

# Select the device
DEVICE = 'cuda'
NUM_GPU = 1
if DEVICE == 'cuda':
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'mps':
    accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    accelerator = 'cpu'
# Set the random seed
torch.manual_seed(42)

# %% Define DataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from torch_extend.datamodule.detection.coco import CocoDataModule
from torch_extend.display.detection import show_bounding_boxes

# Preprocessing
transforms = A.Compose([
    A.Resize(640, 640),  # Resize the image to (640, 640)
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalization (mean and std of the imagenet dataset for normalizing)
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# from torch_extend.dataset.detection.coco import CocoDetectionTV
# from torch.utils.data import DataLoader
# train_dataset = CocoDetectionTV(
#             f'{DATA_ROOT}/train2017',
#             annFile=f'{DATA_ROOT}/annotations/instances_train2017.json', 
#             transforms=transforms
#         )
# trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_iter = iter(trainloader)
# images, targets = next(train_iter)

# Datamodule
datamodule = CocoDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATA_ROOT,
                            transforms=transforms)
datamodule.setup()

# Display the first minibatch
train_iter = iter(datamodule.train_dataloader())
imgs, targets = next(train_iter)
for i, (img, target) in enumerate(zip(imgs, targets)):
    img = (img*255).to(torch.uint8)  # Change from float[0, 1] to uint[0, 255]
    boxes, labels = target['boxes'], target['labels']
    show_bounding_boxes(img, boxes, labels=labels, idx_to_class=datamodule.train_dataset.idx_to_class)
    plt.show()

# %% Create PyTorch Lightning module
from torch_extend.lightning.detection.faster_rcnn import FasterRCNNModule

model = FasterRCNNModule(class_to_idx=datamodule.train_dataset.class_to_idx)

# %% Training
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

# CSV logger
logger = CSVLogger(save_dir='./results/log',
                   name='faster_rcnn', version=0)
trainer = Trainer(accelerator, devices=NUM_GPU, logger=logger)
trainer.fit(model, datamodule=datamodule)

# %%
