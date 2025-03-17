import albumentations as A
from torchvision.transforms import v2
from transformers import BaseImageProcessor

def validate_same_img_size(transforms: v2.Compose | A.Compose,
                           processor: BaseImageProcessor = None) -> bool:
    """Check if the transforms resize/crop the images to the same size"""
    if transforms is None:
        return False
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
    if processor is not None:
        if processor.do_resize and 'height' in processor.size.keys() and 'width' in processor.size.keys():
            return True
    return False
