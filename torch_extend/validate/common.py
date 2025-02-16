import albumentations as A
from torchvision.transforms import v2

def validate_same_img_size(transforms):
    """Check if the transforms resize/crop the images to the same size"""
    for t in transforms.transforms:
        if isinstance(t, A.Resize) or \
           isinstance(t, A.RandomCrop) or \
           isinstance(t, A.CenterCrop) or \
           isinstance(t, A.AtLeastOneBBoxRandomCrop) or \
           isinstance(t, A.RandomSizedBBoxSafeCrop) or \
           isinstance(t, A.CropNonEmptyMaskIfExists):
            if t.height == t.width:
                return True
        if isinstance(t, A.RandomResizedCrop) and t.size[0] == t.size[1]:
            return True
        if isinstance(t, v2.Resize) or \
           isinstance(t, v2.RandomCrop) or \
           isinstance(t, v2.CenterCrop) or \
           isinstance(t, v2.FiveCrop) or \
           isinstance(t, v2.TenCrop) or \
           isinstance(t, v2.RandomResizedCrop):
            if t.size[0] == t.size[1]:
                return True
    return False
