import albumentations as A
from torchvision.transforms import v2
from torchvision.datasets.vision import StandardTransform
from transformers import BaseImageProcessor

# Denormalize the image
def denormalize_image(img,
                      transforms: StandardTransform | A.Compose,
                      processor: BaseImageProcessor = None):
    # Denormalization based on the transforms
    transform_list = transforms.transforms if isinstance(transforms, A.Compose) else transforms.transform.transforms
    for tr in transform_list:
        if isinstance(tr, v2.Normalize) or isinstance(tr, A.Normalize):
            reverse_transform = v2.Compose([
                v2.Normalize(mean=[-mean/std for mean, std in zip(tr.mean, tr.std)],
                                    std=[1/std for std in tr.std])
            ])
            img = reverse_transform(img)
    # Denormalization based on the processor or Transformers
    if processor is not None and processor.do_normalize:
        denormalize_image = v2.Compose([
            v2.Normalize(mean=[-mean/std for mean, std in zip(processor.image_mean, processor.image_std)],
                        std=[1/std for std in processor.image_std])
        ])
        img = denormalize_image(img)
    return img
