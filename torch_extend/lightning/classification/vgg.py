import torch.nn as nn
from torchvision import models

from torch_extend.lightning.detection.base_detection import DetectionModule

class VGGModule(DetectionModule):

    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 first_epoch_lr_scheduled=True,
                 opt_name='sgd', lr=0.02, momentum=0.9, weight_decay=1e-4,
                 lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1,
                 epochs=None, n_batches=None,
                 model_name='vgg11', dropout=0.5):
        super().__init__(class_to_idx,
                         criterion, pretrained, tuned_layers, first_epoch_lr_scheduled,
                         opt_name, lr, momentum, weight_decay, lr_scheduler, lr_step_size, lr_steps, lr_gamma,
                         epochs, n_batches)
        self.model_name = model_name
        self.model: models.VGG
        # Model parameters
        self.dropout = dropout

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load FasterRCNN model based on the `model_name`"""
        if self.model_name == 'vgg11':
            model = models.vgg11(pretrained=models.VGG11_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg11_bn':
            model = models.vgg11_bn(pretrained=models.VGG11_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg13':
            model = models.vgg13(pretrained=models.VGG13_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg13_bn':
            model = models.vgg13_bn(pretrained=models.VGG13_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=models.VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=models.VGG16_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg19':
            model = models.vgg19(pretrained=models.VGG19_Weights.IMAGENET1K_V1 if self.pretrained else None)
        elif self.model_name == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=models.VGG19_BN_Weights.IMAGENET1K_V1 if self.pretrained else None)
        return model

    @property
    def default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Replace the classifier
        num_classes = max(self.class_to_idx.values()) + 1
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, num_classes),
        )
        # Initialize the weights
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
