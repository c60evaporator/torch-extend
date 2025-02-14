from pycocotools import mask as coco_mask
import numpy as np

class SemanticOutput():
    def _get_images_targets(self):
        sample_size = self.__len__()
        images_targets = [self.__getitem__(idx) for idx in range(sample_size)]
        images = [image for image, target in images_targets]
        targets = [target for image, target in images_targets]
        return images, targets