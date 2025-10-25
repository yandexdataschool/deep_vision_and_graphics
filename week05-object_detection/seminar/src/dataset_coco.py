import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2


class COCODetectionDataset(Dataset):
    def __init__(self, root_dir, split='val', target_size=(640, 640), min_area=100, filter_categories=None):
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.min_area = min_area
        self.filter_categories = filter_categories
        
        if split in ['train', 'test', 'val', 'mini_train', 'unlabeled']:
            ann_file = os.path.join(root_dir, 'annotations', f'instances_val2017_{split}.json')
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'test', 'val', 'mini_train', or 'unlabeled'.")
        self.coco = COCO(ann_file)
        
        if filter_categories:
            self.cat_ids = [cat_id for cat_id in self.coco.getCatIds() if self.coco.loadCats(cat_id)[0]['name'] in filter_categories]
        else:
            self.cat_ids = self.coco.getCatIds()
        
        self.cat_id_to_continuous_id = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        
        self.img_ids = []
        for img_id in self.coco.imgs.keys():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)
        
        categories = self.coco.loadCats(self.cat_ids)
        self.class_names = [cat['name'] for cat in categories]
        
        if split in ['train', 'mini_train']:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
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
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_visibility=0.3))
        else:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        
        print(f'Loaded {len(self.img_ids)} images with annotations for {split} split')
        print(f'Classes: {len(self.class_names)} ({", ".join(self.class_names)})')
        if split in ['train', 'mini_train']:
            print(f'Augmentations: Enabled (HorizontalFlip, ShiftScaleRotate, Brightness/Contrast, HSV, GaussNoise)')
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, 'val2017', img_info['file_name'])
        
        image = Image.open(img_path).convert('RGB')
        original_w, original_h = image.size
        image = np.array(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        category_ids = []
        
        for ann in anns:
            if ann['area'] < self.min_area:
                continue
            
            bbox = ann['bbox']
            category_ids.append(self.cat_id_to_continuous_id[ann['category_id']])
            bboxes.append(bbox)
        
        if len(bboxes) > 0:
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed['image']
            bboxes = transformed['bboxes']
            category_ids = transformed['category_ids']
        else:
            transformed = self.transform(image=image, bboxes=[], category_ids=[])
            image = transformed['image']
            bboxes = []
            category_ids = []
        
        if len(bboxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_normalized = []
            for bbox in bboxes:
                x, y, w, h = bbox
                cx = (x + w / 2) / self.target_size[1]
                cy = (y + h / 2) / self.target_size[0]
                w_norm = w / self.target_size[1]
                h_norm = h / self.target_size[0]
                boxes_normalized.append([cx, cy, w_norm, h_norm])
            
            boxes_tensor = torch.tensor(boxes_normalized, dtype=torch.float32)
            labels_tensor = torch.tensor(category_ids, dtype=torch.long)
        
        return image, boxes_tensor, labels_tensor
    
    def get_category_info(self):
        categories = self.coco.loadCats(self.cat_ids)
        return {cat['id']: cat['name'] for cat in categories}

