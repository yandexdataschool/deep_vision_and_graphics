from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import JaccardIndex, MeanMetric


class PersonSegModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        ignore_index: int = 255,
        max_lr_pct_start: float = 0.3,
    ) -> None:
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer
        self.ignore_index = ignore_index
        self.max_lr_pct_start = max_lr_pct_start

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_iou = JaccardIndex(task="binary", num_classes=2)
        self.test_iou = JaccardIndex(task="binary", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _combine_inputs(self, batch) -> torch.Tensor:
        image = batch["image"].float()
        depth = batch.get("depth")
        if depth is not None:
            return torch.cat([image, depth.float()], dim=1)
        return image

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs = self._combine_inputs(batch)
        masks = batch["mask"]

        logits = self(inputs)
        loss = self.criterion(logits, masks)

        self.train_loss.update(loss.detach())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=inputs.size(0))

        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.train_loss.compute()
        self.log("train_epoch/loss", loss, prog_bar=True)
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs = self._combine_inputs(batch)
        masks = batch["mask"]

        logits = self(inputs)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.val_loss.update(loss.detach())
        self.val_iou.update(preds, masks)

        return loss

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute()
        iou = self.val_iou.compute()

        self.log("val/loss", loss, prog_bar=False)
        self.log("val/mIoU", iou, prog_bar=True)

        self.val_loss.reset()
        self.val_iou.reset()

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs = self._combine_inputs(batch)
        masks = batch["mask"]

        logits = self(inputs)
        loss = self.criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        self.test_loss.update(loss.detach())
        self.test_iou.update(preds, masks)

        return loss

    def on_test_epoch_end(self) -> None:
        loss = self.test_loss.compute()
        iou = self.test_iou.compute()

        self.log("test/loss", loss, prog_bar=False)
        self.log("test/mIoU", iou, prog_bar=True)

        self.test_loss.reset()
        self.test_iou.reset()

    def configure_optimizers(self):
        if self.optimizer_type.lower() == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        total_steps = max(1, getattr(self.trainer, "estimated_stepping_batches", 1))
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=self.max_lr_pct_start,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
