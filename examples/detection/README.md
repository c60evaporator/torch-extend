

## Tips

### Hyperpameters

#### 

#### Learning rate


## Example

### VOC detection dataset

If you follow [this repository](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/tree/master), please use the following Hyperparameters and Transforms.

```python
###### Hyperparameters (Irrelevant parameters are omitted) ######
# General Parameters
EPOCHS = 14
BATCH_SIZE = 1
NUM_WORKERS = 8
# Optimizer Parameter
OPT_NAME = 'sgd'
LR = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9  # For SGD and RMSprop
# LR Scheduler Parameters
LR_SCHEDULER = 'steplr'
LR_GAMMA = 0.1
LR_STEP_SIZE = 9  # For StepLR


###### Transforms ######
import albumentations as A
from albumentations.pytorch import ToTensorV2

NORM_MEAN = [0.0, 0.0, 0.0]
NORM_STD = [1.0, 1.0, 1.0]
# Transforms for training (https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/data/dataset.py)
def edge_resize_image(image, **kwargs):
    """Short edge and long edge threshold resize (https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/data/dataset.py#L42)"""
    SHORT_EDGE_THRESH = 600
    LONG_EDGE_THRESH = 1000
    scale1 = SHORT_EDGE_THRESH / min(kwargs['shape'][:2])
    scale2 = LONG_EDGE_THRESH / max(kwargs['shape'][:2])
    scale = min(scale1, scale2)
    return A.Resize(height=int(kwargs['shape'][0] * scale), width=int(kwargs['shape'][1] * scale))(image=image)['image']

def edge_resize_bboxes(src, **kwargs):
    return src  # Do nothing because bbox is relative to the image size

train_transform = A.Compose([
    A.Lambda(image=edge_resize_image, bboxes=edge_resize_bboxes),
    A.HorizontalFlip(p=0.5),
    A.Normalize(NORM_MEAN, NORM_STD),  # Normalization (mean and std of the imagenet dataset for normalizing)
    ToTensorV2()  # Convert from range [0, 255] to a torch.FloatTensor in the range [0.0, 1.0]
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Transforms for training
eval_transform = A.Compose([
    A.Normalize(NORM_MEAN, NORM_STD),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
```