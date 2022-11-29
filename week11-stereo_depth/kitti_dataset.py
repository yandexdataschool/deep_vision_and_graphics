import os

import numpy as np
from PIL import Image
import skimage
import skimage.io
from torch.utils.data import Dataset
from tqdm import tqdm


def read_rgb_image(path_to_image):
    image = skimage.img_as_ubyte(skimage.io.imread(path_to_image))
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    if image.shape[-1] == 1:
        image = image * np.ones((1,1,3), dtype=np.uint8)
    return image


class KITTIStereoRAM(Dataset):
    def __init__(self, root, train=True, transforms=None):
        super(KITTIStereoRAM, self).__init__()

        self.root = root
        if train:
            path_to_dataset = os.path.join(root, 'train')
        else:
            path_to_dataset = os.path.join(root, 'val')

        self.transforms = transforms
        self.left_images = []
        self.right_images = []
        self.targets = []
        self.valid_pixels_masks = []
        for img_name in tqdm(os.listdir(os.path.join(path_to_dataset, 'colored_0'))):
            left_img_path = os.path.join(path_to_dataset, 'colored_0', img_name)
            right_img_path = os.path.join(path_to_dataset, 'colored_1', img_name)
            disp_img_path = os.path.join(path_to_dataset, 'disp_noc', img_name)

            left_img = read_rgb_image(left_img_path)
            right_img = read_rgb_image(right_img_path)

            # disparity normalization
            disp_img = skimage.io.imread(disp_img_path)[:,:,np.newaxis]
            valid_pixels_mask = disp_img > 0
            disp_img = disp_img.astype(np.float32) / 256.

            self.left_images.append(Image.fromarray(left_img))
            self.right_images.append(Image.fromarray(right_img))
            self.targets.append(disp_img)
            self.valid_pixels_masks.append(valid_pixels_mask)

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, index):
        left_img = self.left_images[index]
        right_img = self.right_images[index]
        disp = self.targets[index]
        valid_pixels_mask = self.valid_pixels_masks[index]

        if self.transforms is not None:
            left_img, right_img, disp, valid_pixels_mask = self.transforms(
                    left_img, right_img, disp, valid_pixels_mask)
        return left_img, right_img, disp, valid_pixels_mask
