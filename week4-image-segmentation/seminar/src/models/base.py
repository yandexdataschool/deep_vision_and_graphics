import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSegmentationModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def compute_loss(self, output, target):
        pass
    
    @abstractmethod
    def postprocess(self, output):
        pass

