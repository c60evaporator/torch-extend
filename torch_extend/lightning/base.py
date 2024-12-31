import torch
import lightning.pytorch as pl

from abc import ABC, abstractmethod

class TorchVisionModule(pl.LightningModule, ABC):
    def __init__(self, model_name, criterion=None,
                 pretrained=False, tuned_layers=None,
                 first_epoch_lr_scheduled = False):
        super().__init__()
        self.model_name = model_name
        # Save the criterion
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = self.default_criterion

        # Pretraining configuration
        self.pretrained = pretrained
        if tuned_layers is not None:
            self.tuned_layers = tuned_layers
        else:
            self.tuned_layers = self.default_tuned_layers

        # first_epoch_lr_scheduler (https://github.com/pytorch/vision/blob/main/references/detection/engine.py)
        self.first_epoch_lr_scheduled = first_epoch_lr_scheduled
        self.first_epoch_lr_scheduler: torch.optim.lr_scheduler.LinearLR = None

        # Other
        self.model = None
        self.schedulers = None
    
    ###### Set the model and the fine-tuning settings ######
    @abstractmethod
    def _get_model(self) -> torch.nn.Module:
        """Default model"""
        raise NotImplementedError
    
    @property
    def default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []

    def replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        pass

    def _set_model_and_params(self) -> torch.nn.Module:
        # Set the model
        self.model = self._get_model()
        # Fine tuning setting
        if self.pretrained:
            for name, param in self.model.named_parameters():
                if name in self.tuned_layers:  # Fine tuned layers
                    param.requires_grad = True
                else:  # No-tuned layers
                    param.requires_grad = False
            # Transfer learning setting
            self.replace_transferred_layers()

    def _setup(self):
        pass
    
    def setup(self, stage: str | None = None):
        # Set the model and its parameters
        self._set_model_and_params()
        
        # For logging
        self.train_epoch_losses = []
        self.train_running_loss = 0.0
        self.i_epoch = 0

        # Additional processes during the setup
        self._setup()

    ###### Training ######
    @abstractmethod
    def default_criterion(self):
        """Default criterion"""
        raise NotImplementedError
    
    @abstractmethod
    def calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        """Training"""
        loss = self.calc_train_loss(batch)
        # Record the loss
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_running_loss += loss.item()
        self.last_batch = batch_idx
        # first_epoch_lr_scheduler
        if self.first_epoch_lr_scheduler is not None:
            self.first_epoch_lr_scheduler.step()
        return loss
    
    def on_train_epoch_end(self):
        """Epoch end processes during the training"""
        # Record the epoch loss
        self.train_epoch_losses.append(self.train_running_loss / (self.last_batch+1))
        self.train_running_loss = 0.0
        # Disable first_epoch_lr_scheduler
        self.first_epoch_lr_scheduler = None
        # Scheduler step
        for scheduler in self.schedulers:
            scheduler.step()
    
    ###### Validation ######
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        """Epoch end processes during the validation"""
        # Increment the epoch
        self.i_epoch += 1

    ###### Test ######
    @abstractmethod
    def test_step(self, batch, batch_idx):
        loss = self.calc_val_loss(batch)
        # Record the loss
        self.log("test_loss", loss.item())

    ###### Prediction ######
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
    
    ##### Optimizers and Schedulers ######
    def configure_optimizers(self):
        raise NotImplementedError
