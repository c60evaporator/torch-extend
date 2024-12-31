#%% Select the device
import os
import sys
# Add the root directory of the repository to system pathes (For debugging)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import torch

# Parameters
EPOCHS = 1
BATCH_SIZE = 8
NUM_WORKERS = 4
DATA_ROOT = '/home/knakamura/Programs/Python/torch-benchmarks/object_detection/datasets/COCO'

# Select the device
DEVICE = 'cuda'
if DEVICE == 'cuda':
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
elif DEVICE == 'mps':
    accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
else:
    accelerator = 'cpu'
# Set the random seed
torch.manual_seed(42)
# Multi GPU (https://github.com/pytorch/pytorch/issues/40403)
NUM_GPU = 1

# %% Define DataModule
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch_extend.datamodule.detection.coco import CocoDataModule

# Preprocessing
transforms = A.Compose([
    A.Resize(640, 640),  # Resize the image to (640, 640)
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalization (mean and std of the imagenet dataset for normalizing)
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# Datamodule
datamodule = CocoDataModule(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, root=DATA_ROOT,
                            train_annFile='./ann_validation/COCO/instances_train_filtered.json',
                            val_annFile='./ann_validation/COCO/instances_val_filtered.json',
                            transforms=transforms)
datamodule.setup()
#datamodule.validate_annotation(use_instance_loader=True)

# Display the first minibatch
datamodule.show_first_minibatch(image_set='train')

# %% Create PyTorch Lightning module
from torch_extend.lightning.detection.faster_rcnn import FasterRCNNModule

model = FasterRCNNModule(class_to_idx=datamodule.class_to_idx)

# %% Training
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

# CSV logger
logger = CSVLogger(save_dir=f'./log/{datamodule.dataset_name}/{model.model_name}',
                   name=model.model_weight, version=0)
trainer = Trainer(accelerator, devices=NUM_GPU, max_epochs=EPOCHS, logger=logger)
trainer.fit(model, datamodule=datamodule)
        
# %%
