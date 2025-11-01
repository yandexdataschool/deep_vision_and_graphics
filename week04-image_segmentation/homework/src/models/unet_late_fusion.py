from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base import BaseSegmentationModel

__all__ = ["UNetResNet50LateFusion"]


class UNetResNet50LateFusion(BaseSegmentationModel):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True
    ) -> None:
        super().__init__()
        # TODO

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        # TODO
