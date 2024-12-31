import torch
from abc import abstractmethod
import numpy as np

from ..base import TorchVisionModule
from ...metrics.detection import average_precisions

class DetectionModule(TorchVisionModule):
    def __init__(self, class_to_idx: dict[str, int],
                 model_name, criterion=None,
                 pretrained=True, tuned_layers=None,
                 first_epoch_lr_scheduled=True,
                 opt_name='sgd', lr=0.02, momentum=0.9, weight_decay=1e-4,
                 lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1,
                 epochs=None, n_batches=None):
        super().__init__(model_name, criterion, pretrained, tuned_layers, first_epoch_lr_scheduled)
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
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma
        # Set the number of epochs and batches for lr_scheduler
        self.epochs = epochs
        self.n_batches = n_batches  # for first_epoch_lr_scheduler

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
        self.val_step_targets = []
        self.val_step_preds = []
        self.test_step_targets = []
        self.test_step_preds = []
    
    ###### Validation ######
    def _get_preds_targets(self, batch):
        inputs, targets = batch
        preds = self.model(inputs)
        preds_cpu = [{k: v.cpu() for k, v in pred.items()} for pred in preds]
        targets_cpu = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                       for target in targets]
        return preds_cpu, targets_cpu

    def validation_step(self, batch, batch_idx):
        # Store the predictions
        preds_cpu, targets_cpu = self._get_preds_targets(batch)
        self.val_step_preds.extend(preds_cpu)
        self.val_step_targets.extend(targets_cpu)

    def on_validation_epoch_end(self):
        """Epoch end processes during the validation"""
        # Calculate Average Precision
        aps = average_precisions(self.val_step_preds, self.val_step_targets,
                                self.idx_to_class, 
                                iou_threshold=0.5, conf_threshold=0.0)
        mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
        print(f'mAP={mean_average_precision}')
        self.val_step_targets = []
        self.val_step_preds = []
        # Increment the epoch
        self.i_epoch += 1

    ###### Test ######
    def test_step(self, batch, batch_idx):
        # Store the predictions
        preds_cpu, targets_cpu = self._get_preds_targets(batch)
        self.test_step_preds.extend(preds_cpu)
        self.test_step_targets.extend(targets_cpu)

    def on_test_epoch_end(self):
        """Epoch end processes during the test"""
        # Index to class names dict with background
        idx_to_class_bg = {k: v for k, v in self.idx_to_class.items()}
        idx_to_class_bg[-1] = 'background'
        # Calculate Average Precision
        aps = average_precisions(self.test_step_preds, self.test_step_targets,
                                 idx_to_class_bg,
                                 iou_threshold=0.5, conf_threshold=0.0)
        mean_average_precision = np.mean([v['average_precision'] for v in aps.values()])
        print(f'mAP={mean_average_precision}')
        self.test_step_targets = []
        self.test_step_preds = []
        # Increment the epoch
        self.i_epoch += 1
    
    ##### Optimizers and Schedulers ######
    def configure_optimizers(self):
        """Configure optimizers and LR schedulers"""
        # Optimizer (Reference https://github.com/pytorch/vision/blob/main/references/detection/train.py)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov="nesterov" in self.opt_name,
            )
        elif self.opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise RuntimeError(f'Invalid optimizer {self.opt_name}. Only "sgd", "sgd_nesterov", and "adamw" are supported.')

        # lr_schedulers
        self.schedulers = []
        if self.lr_scheduler == "multisteplr":
            self.schedulers.append(torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_steps, gamma=self.lr_gamma))
        elif self.lr_scheduler == "cosineannealinglr":
            if self.epochs is None:
                raise RuntimeError(f'The `epochs` argument should be specified if "cosineannealinglr" is selected as the lr_scheduler.')
            self.schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs))
        else:
            raise RuntimeError(f'Invalid lr_scheduler {self.lr_scheduler}. Only "multisteplr" and "cosineannealinglr" are supported.')

        # first_epoch_lr_scheduler (https://github.com/pytorch/vision/blob/main/references/detection/engine.py)
        if self.first_epoch_lr_scheduled:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, self.n_batches - 1) if self.n_batches is not None else 1000
            self.first_epoch_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )

        return optimizer
