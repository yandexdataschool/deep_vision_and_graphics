import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import MeanMetric


class DetectionModule(pl.LightningModule):
    def __init__(
        self,
        model=None,
        num_classes: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = 'adamw'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        
        self.train_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_map = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, boxes, labels = batch
        
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b],
                'labels': labels[b]
            })
        
        outputs = self(images)
        loss = self.model.compute_loss(outputs, targets)
        
        self.train_loss.update(loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, boxes, labels = batch
        
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b],
                'labels': labels[b]
            })
        
        outputs = self(images)
        loss = self.model.compute_loss(outputs, targets)
        
        preds = self.model.postprocess(outputs, conf_threshold=0.1)
        
        preds_for_metric = []
        targets_for_metric = []
        
        for b in range(len(boxes)):
            preds_for_metric.append({
                'boxes': preds[b]['boxes'],
                'scores': preds[b]['scores'],
                'labels': preds[b]['labels']
            })
            targets_for_metric.append({
                'boxes': targets[b]['boxes'],
                'labels': targets[b]['labels']
            })
        
        self.val_loss.update(loss)
        self.val_map.update(preds_for_metric, targets_for_metric)
        
        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        
        self.log('train_loss', train_loss, prog_bar=False)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=False)
        
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
        
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_map_dict = self.val_map.compute()
        val_map = val_map_dict['map']
        val_map50 = val_map_dict['map_50']
        
        self.log('val_loss', val_loss, prog_bar=False)
        self.log('val_map', val_map, prog_bar=True)
        self.log('val_map50', val_map50, prog_bar=False)
        
        print(f"[Epoch {self.current_epoch}] Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f} | Val mAP@50: {val_map50:.4f}")
        
        self.val_loss.reset()
        self.val_map.reset()
    
    def test_step(self, batch, batch_idx):
        images, boxes, labels = batch
        
        targets = []
        for b in range(len(boxes)):
            targets.append({
                'boxes': boxes[b],
                'labels': labels[b]
            })
        
        outputs = self(images)
        loss = self.model.compute_loss(outputs, targets)
        
        preds = self.model.postprocess(outputs, conf_threshold=0.1)
        
        preds_for_metric = []
        targets_for_metric = []
        
        for b in range(len(boxes)):
            preds_for_metric.append({
                'boxes': preds[b]['boxes'],
                'scores': preds[b]['scores'],
                'labels': preds[b]['labels']
            })
            targets_for_metric.append({
                'boxes': targets[b]['boxes'],
                'labels': targets[b]['labels']
            })
        
        self.val_loss.update(loss)
        self.val_map.update(preds_for_metric, targets_for_metric)
        
        return loss
    
    def on_test_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_map_dict = self.val_map.compute()
        val_map = val_map_dict['map']
        val_map50 = val_map_dict['map_50']
        
        self.log('test_loss', val_loss, prog_bar=False)
        self.log('test_map', val_map, prog_bar=True)
        self.log('test_map50', val_map50, prog_bar=False)
        
        print(f"\n[Test] Loss: {val_loss:.4f} | mAP: {val_map:.4f} | mAP@50: {val_map50:.4f}")
        
        self.val_loss.reset()
        self.val_map.reset()
    
    def configure_optimizers(self):
        if self.optimizer_type == 'adamw':
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

