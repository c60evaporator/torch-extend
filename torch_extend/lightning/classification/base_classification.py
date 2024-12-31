import torch
import torch.nn as nn
from abc import abstractmethod

from ..base import TorchVisionModule

class ClassificationModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 opt_name='sgd', lr=0.1, momentum=0.9, weight_decay=1e-4,
                 lr_scheduler='steplr', lr_step_size=30, lr_gamma=0.1,
                 epochs = None):
        super().__init__(model_name, criterion, pretrained, tuned_layers, False)
        # Class to index dict
        self.class_to_idx = class_to_idx
        # Index to class dict
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        na_cnt = 0
        for i in range(max(class_to_idx.values())):
            if i not in class_to_idx.values():
                na_cnt += 1
                self.idx_to_class[i] = f'NA{"{:02}".format(na_cnt)}'
        
        # Optimizer parameters
        self.opt_name = opt_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # Learning scheduler parameters
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        # Set the number of epochs and batches for lr_scheduler
        self.epochs = epochs

    ###### Set the model and the fine-tuning settings ######
    @property
    @abstractmethod
    def default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        raise NotImplementedError
    
    @abstractmethod
    def replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        raise NotImplementedError
    
    def _setup(self):
        # For logging
        self.val_epoch_losses = []
        self.val_running_loss = 0.0

    ###### Training ######
    def calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        outputs = self.model(inputs)
        return self.criterion(outputs, targets)
    
    @property
    def default_criterion(self):
        """Default criterion (Sum of all the losses)"""
        return nn.CrossEntropyLoss()
    
    ###### Validation ######
    def validation_step(self, batch, batch_idx):
        loss = self.calc_train_loss(batch)
        # Record the loss
        self.log("val_loss", loss.item())
        self.val_running_loss += loss.item()
        self.last_batch = batch_idx

    def on_validation_epoch_end(self):
        """Epoch end processes during the validation"""
        self.val_epoch_losses.append(self.val_running_loss / (self.last_batch+1))
        self.val_running_loss = 0.0
        # Increment the epoch
        self.i_epoch += 1

    ###### Test ######
    def test_step(self, batch, batch_idx):
        loss = self.calc_train_loss(batch)
        # Record the loss
        self.log("test_loss", loss.item())
    
    ##### Optimizers and Schedulers ######
    def configure_optimizers(self):
        """Configure optimizers and LR schedulers"""
        # Optimizer (Reference https://github.com/pytorch/vision/blob/main/references/classification/train.py)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov="nesterov" in self.opt_name,
            )
        elif self.opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay, eps=0.0316, alpha=0.9
            )
        elif self.opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise RuntimeError(f'Invalid optimizer {self.opt_name}. Only "sgd", "sgd_nesterov", "rmsprop", and "adamw" are supported.')
    
        # lr_schedulers
        self.schedulers = []
        if self.lr_scheduler == "steplr":
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma))
        elif self.lr_scheduler == "cosineannealinglr":
            if self.epochs is None:
                raise RuntimeError(f'The `epochs` argument should be specified if "cosineannealinglr" is selected as the lr_scheduler.')
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs))
        elif self.lr_scheduler == "exponentiallr":
            self.schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma))
        else:
            raise RuntimeError(f'Invalid lr_scheduler {self.lr_scheduler}. Only "steplr", "cosineannealinglr", and "exponentiallr" are supported.')

        return optimizer
