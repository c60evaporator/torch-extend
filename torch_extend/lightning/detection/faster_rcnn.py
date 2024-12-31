from torchvision.models.detection import faster_rcnn

from .base_detection import DetectionModule

class FasterRCNNModule(DetectionModule):
    def __init__(self, class_to_idx,
                 criterion=None,
                 pretrained=True, tuned_layers=None,
                 first_epoch_lr_scheduled=True,
                 opt_name='sgd', lr=0.02, momentum=0.9, weight_decay=1e-4,
                 lr_scheduler='multisteplr', lr_step_size=8, lr_steps=[16, 22], lr_gamma=0.1,
                 epochs=None, n_batches=None,
                 model_weight='fasterrcnn_resnet50_fpn_v2'):
        super().__init__(class_to_idx,
                         'fasterrcnn',
                         criterion, pretrained, tuned_layers, first_epoch_lr_scheduled,
                         opt_name, lr, momentum, weight_decay, lr_scheduler, lr_step_size, lr_steps, lr_gamma,
                         epochs, n_batches)
        self.model_weight = model_weight
        self.model: faster_rcnn.FasterRCNN

    ###### Set the model and the fine-tuning settings ######
    def _get_model(self):
        """Load FasterRCNN model based on the `model_weight`"""
        if self.model_weight == 'fasterrcnn_resnet50_fpn':
            model = faster_rcnn.fasterrcnn_resnet50_fpn(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_resnet50_fpn_v2':
            model = faster_rcnn.fasterrcnn_resnet50_fpn_v2(weights=faster_rcnn.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_mobilenet_v3_large_320_fpn':
            model = faster_rcnn.fasterrcnn_mobilenet_v3_large_320_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if self.pretrained else None)
        elif self.model_weight == 'fasterrcnn_mobilenet_v3_large_fpn':
            model = faster_rcnn.fasterrcnn_mobilenet_v3_large_fpn(weights=faster_rcnn.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if self.pretrained else None)
        else:
            raise RuntimeError(f'Invalid `model_weight` {self.model_weight}.')
        return model

    @property
    def default_tuned_layers(self) -> list[str]:
        """Layers subject to the fine tuning"""
        return []
    
    def replace_transferred_layers(self) -> None:
        """Replace layers for transfer learning"""
        # Replace the box_predictor
        num_classes = max(self.class_to_idx.values()) + 1
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    ###### Training ######
    def calc_train_loss(self, batch):
        """Calculate the training loss from the batch"""
        inputs, targets = batch
        loss_dict = self.model(inputs, targets)
        return self.criterion(loss_dict)
    
    def default_criterion(self, outputs):
        """Default criterion (Sum of all the losses)"""
        return sum(loss for loss in outputs.values())
