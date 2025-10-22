import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCOInstanceDataset(Dataset):
    def __init__(self, root_dir, split='val', target_size=(360, 480), min_area=100, mode='instance'):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.min_area = min_area
        self.mode = mode
        
        if split in ['train', 'test', 'val', 'mini_train', 'unlabeled']:
            ann_file = os.path.join(root_dir, 'annotations', f'instances_val2017_{split}.json')
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', 'val', 'mini_train', or 'unlabeled'.")
        self.coco = COCO(ann_file)
        
        self.img_ids = list(self.coco.imgs.keys())
        if split != 'unlabeled':
            self.img_ids = [img_id for img_id in self.img_ids 
                           if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_continuous_id = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.cat_id_to_continuous_id[0] = 0
        
        categories = self.coco.loadCats(cat_ids)
        self.class_names = ['background'] + [cat['name'] for cat in categories]
        
        if split in ['train', 'mini_train']:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, 
                    scale_limit=0.15, 
                    rotate_limit=15, 
                    border_mode=0,
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        print(f'Loaded {len(self.img_ids)} images with annotations for {split} split (mode: {mode})')
        print(f'Classes: {len(self.class_names)} (0=background + {len(cat_ids)} COCO categories)')
        if split in ['train', 'mini_train']:
            print(f'Augmentations: Enabled (HorizontalFlip, ShiftScaleRotate, Brightness/Contrast, HSV, GaussNoise)')
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, 'val2017', img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        image = np.array(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        instance_masks = []
        instance_labels = []
        
        for ann in anns:
            if ann.get('iscrowd', 0) == 1:
                continue
            
            if ann['area'] < self.min_area:
                continue
            
            mask = self.coco.annToMask(ann)
            
            if mask.sum() == 0:
                continue
            
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask = np.array(mask) > 0
            
            instance_masks.append(mask.astype(np.uint8))
            continuous_id = self.cat_id_to_continuous_id[ann['category_id']]
            instance_labels.append(continuous_id)
        
        if len(instance_masks) > 0:
            transformed = self.transform(image=image, masks=instance_masks)
            image = transformed['image']
            instance_masks = transformed['masks']
            
            valid_indices = []
            min_pixels_after_aug = 50
            for i, mask in enumerate(instance_masks):
                mask_np = np.asarray(mask)
                mask_area = mask_np.sum()
                if mask_area >= min_pixels_after_aug:
                    valid_indices.append(i)
            
            instance_masks = [instance_masks[i] for i in valid_indices]
            instance_labels = [instance_labels[i] for i in valid_indices]
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        if self.mode == 'semantic':
            semantic_mask = np.zeros(self.target_size, dtype=np.int64)
            
            if len(instance_masks) > 0:
                instance_masks_np = [np.asarray(mask) > 0 for mask in instance_masks]
                areas = [mask.sum() for mask in instance_masks_np]
                sorted_indices = np.argsort(areas)[::-1]
                
                for idx in sorted_indices:
                    mask = instance_masks_np[idx]
                    label = instance_labels[idx]
                    semantic_mask[mask] = label
            
            return image, torch.from_numpy(semantic_mask)
        
        else:
            if len(instance_masks) == 0:
                instance_masks = np.zeros((1, self.target_size[0], self.target_size[1]), dtype=np.float32)
                instance_labels = [0]
            else:
                instance_masks = [np.asarray(mask).astype(np.float32) for mask in instance_masks]
                instance_masks = np.stack(instance_masks, axis=0)
            
            instance_masks = torch.from_numpy(instance_masks)
            instance_labels = torch.tensor(instance_labels, dtype=torch.long)
            
            return image, instance_masks, instance_labels
    
    def get_category_info(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        return {cat['id']: cat['name'] for cat in categories}

