import os

import numpy as np
from PIL import Image
import skimage
import skimage.io
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def read_rgb_image(path_to_image):
    image = skimage.img_as_ubyte(skimage.io.imread(path_to_image))
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if image.shape[-1] == 1:
        image = image * np.ones((1,1,3), dtype=np.uint8)
    return image


class TinyImagenetRAM(Dataset):
    def __init__(self, root, transform=transforms.ToTensor()):
        super(TinyImagenetRAM, self).__init__()

        self.root = root
        self.classes = sorted(
            [item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item))])
        self.class_to_idx = {item: index for index, item in enumerate(self.classes)}

        self.transform = transform
        self.images, self.targets = [], []
        for index, item in tqdm(enumerate(self.classes), total=len(self.classes), desc=self.root):
            path = os.path.join(root, item, 'images')
            for name in sorted(os.listdir(path)):
                image = read_rgb_image(os.path.join(path, name))
                assert image.shape == (64, 64, 3), image.shape
                self.images.append(Image.fromarray(image))
                self.targets.append(index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        target = self.targets[index]
        return image, target
