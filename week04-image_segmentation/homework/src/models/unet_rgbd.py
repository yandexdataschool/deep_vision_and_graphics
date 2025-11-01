from models.unet import UNetResNet50

__all__ = ["UNetResNet50RGBD"]


class UNetResNet50RGBD(UNetResNet50):

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
        )
        # TODO
