import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import JaccardIndex, MeanMetric


class SegmentationModule(pl.LightningModule):
    def __init__(
        self, 
        model=None,
        num_classes: int = 7, 
        learning_rate: float = 1e-4, 
        weight_decay: float = 1e-5,
        optimizer: str = 'adam'
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.num_classes = num_classes
        self.num_classes_with_bg = num_classes + 1
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        
        self.train_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes + 1, average='macro', ignore_index=0)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, instance_masks, instance_labels = batch
            outputs = self(images)
            targets = self._prepare_targets(instance_masks, instance_labels)
            loss = self.model.compute_loss(outputs, targets)
        else:
            images, masks = batch
            outputs = self(images)
            loss = self.model.compute_loss(outputs, masks)
        
        self.train_loss.update(loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            images, instance_masks, instance_labels = batch
            outputs = self(images)
            targets = self._prepare_targets(instance_masks, instance_labels)
            loss = self.model.compute_loss(outputs, targets)
            
            preds = self.model.postprocess(outputs, mode='semantic')
            semantic_targets = self._instance_to_semantic(instance_masks, instance_labels, device=self.device)
        else:
            images, masks = batch
            outputs = self(images)
            loss = self.model.compute_loss(outputs, masks)
            if hasattr(self.model, 'postprocess'):
                preds = self.model.postprocess(outputs)
            else:
                preds = torch.argmax(outputs, dim=1)
            semantic_targets = masks
        
        self.val_loss.update(loss)
        self.val_iou.update(preds, semantic_targets)
        
        return loss
    
    def _instance_to_semantic(self, instance_masks, instance_labels, device=None):
        B = len(instance_masks)
        H, W = instance_masks[0].shape[-2:]
        
        if device is None:
            device = instance_masks[0].device
        
        semantic_masks = torch.zeros(B, H, W, dtype=torch.long, device=device)
        
        for b in range(B):
            masks_b = instance_masks[b].to(device) if hasattr(instance_masks[b], 'to') else instance_masks[b]
            labels_b = instance_labels[b].to(device) if hasattr(instance_labels[b], 'to') else instance_labels[b]
            
            for i in range(masks_b.shape[0]):
                mask = masks_b[i] > 0.5
                label = labels_b[i].item()
                if label > 0:
                    semantic_masks[b][mask] = label
        
        return semantic_masks
    
    def _prepare_targets(self, instance_masks, instance_labels):
        targets = []
        for b in range(len(instance_masks)):
            valid_idx = instance_labels[b] > 0
            targets.append({
                'labels': instance_labels[b][valid_idx] - 1,
                'masks': instance_masks[b][valid_idx]
            })
        return targets
    
    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        
        self.log('train_loss', train_loss, prog_bar=False)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=False)
        
        print(f"\n[Epoch {self.current_epoch}] Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
        
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        val_iou = self.val_iou.compute()
        
        self.log('val_loss', val_loss, prog_bar=False)
        self.log('val_iou', val_iou, prog_bar=True)
        
        print(f"[Epoch {self.current_epoch}] Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        
        self.val_loss.reset()
        self.val_iou.reset()
    
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
