from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base import BaseSegmentationModel

__all__ = ["UNetResNet50", "UNetResNet50RGB"]


class UNetResNet50(BaseSegmentationModel):
    """Standard UNet decoder on top of a ResNet-50 encoder."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.input_channels = in_channels

        resnet = models.resnet50(pretrained=pretrained)

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec1 = self._make_decoder_block(2048, 1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec2 = self._make_decoder_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._make_decoder_block(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec4 = self._make_decoder_block(128, 64)

        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _make_decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size: Tuple[int, int] = x.shape[2:]

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        bridge = self.bridge(enc5)

        dec1 = self.up1(bridge)
        dec1 = self._match_size(dec1, enc4)
        dec1 = torch.cat([dec1, enc4], dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.up2(dec1)
        dec2 = self._match_size(dec2, enc3)
        dec2 = torch.cat([dec2, enc3], dim=1)
        dec2 = self.dec2(dec2)

        dec3 = self.up3(dec2)
        dec3 = self._match_size(dec3, enc2)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)

        dec4 = self.up4(dec3)
        dec4 = self._match_size(dec4, enc1)
        dec4 = torch.cat([dec4, enc1], dim=1)
        dec4 = self.dec4(dec4)

        dec5 = self.up5(dec4)
        dec5 = self.dec5(dec5)

        output = self.final(dec5)
        output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        return output

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(output, target)

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        return torch.argmax(output, dim=1)


UNetResNet50RGB = UNetResNet50
