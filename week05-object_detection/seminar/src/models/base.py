import torch
import torch.nn as nn


class BaseDetector(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        raise NotImplementedError
    
    def compute_loss(self, outputs, targets):
        raise NotImplementedError
    
    def postprocess(self, outputs, conf_threshold=0.5):
        raise NotImplementedError

