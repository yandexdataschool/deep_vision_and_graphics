from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset

__all__ = ["PersonSegmentationDataset", "build_train_transform", "build_eval_transform"]


def _resolve_additional_targets(include_depth: bool) -> Optional[Dict[str, str]]:
    return {"depth": "image"} if include_depth else None


def build_train_transform(image_size: Tuple[int, int], include_depth: bool) -> A.Compose:
    additional_targets = _resolve_additional_targets(include_depth)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


def build_eval_transform(image_size: Tuple[int, int], include_depth: bool) -> A.Compose:
    additional_targets = _resolve_additional_targets(include_depth)
    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        additional_targets=additional_targets,
    )


class PersonSegmentationDataset(Dataset):
    """COCO person segmentation dataset with optional depth channel."""

    def __init__(
        self,
        data_dir: Path,
        ann_file: Path,
        image_size: Tuple[int, int],
        augment: bool = False,
        depth_dir: Optional[Path] = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.ann_file = Path(ann_file).resolve()
        self.image_size = image_size
        self.depth_dir = Path(depth_dir).resolve() if depth_dir else None

        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        self.coco = COCO(str(self.ann_file))

        person_cat_ids = self.coco.getCatIds(catNms=["person"])
        if not person_cat_ids:
            raise ValueError("COCO annotations do not contain the 'person' category")
        self.person_cat_id = person_cat_ids[0]

        self.img_ids = self.coco.getImgIds(catIds=[self.person_cat_id])

        include_depth = self.depth_dir is not None
        transform_builder = build_train_transform if augment else build_eval_transform
        self.transform = transform_builder(self.image_size, include_depth)

    def __len__(self) -> int:
        return len(self.img_ids)

    # ---------------------------------------------------------------------
    # Loading utilities
    # ---------------------------------------------------------------------
    def _load_image(self, img_id: int) -> Image.Image:
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.data_dir / "val2017" / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        return image

    def _load_mask(self, img_id: int) -> np.ndarray:
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[self.person_cat_id])
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros(self.image_size, dtype=np.uint8)
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue

            ann_mask = self.coco.annToMask(ann)
            if ann_mask.sum() == 0:
                continue

            ann_mask = Image.fromarray(ann_mask.astype(np.uint8) * 255)
            ann_mask = ann_mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)
            ann_mask = np.array(ann_mask, dtype=np.uint8)
            mask[ann_mask > 0] = 1

        return mask

    def _load_depth(self, img_id: int, file_name: str) -> Optional[np.ndarray]:
        if self.depth_dir is None:
            return None

        stem = Path(file_name).stem
        npy_path = self.depth_dir / f"{stem}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Depth file {npy_path} not found. Run the depth prediction script first."
            )

        depth = np.load(npy_path)
        if depth.shape != self.image_size:
            depth_image = Image.fromarray(depth.astype(np.float32), mode="F")
            depth_image = depth_image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
            depth = np.array(depth_image, dtype=np.float32)

        return depth

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]

        image = np.array(self._load_image(img_id))
        mask = self._load_mask(img_id)
        depth = self._load_depth(img_id, img_info["file_name"])

        if depth is not None:
            transformed = self.transform(image=image, mask=mask, depth=depth)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"].long()
            depth_tensor = transformed["depth"]
            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0)
            elif depth_tensor.dim() == 3 and depth_tensor.size(0) == 3:
                depth_tensor = depth_tensor.mean(dim=0, keepdim=True)
            return {
                "image": image_tensor,
                "mask": mask_tensor,
                "depth": depth_tensor.float(),
                "id": img_id,
            }

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].long()
        return {"image": image_tensor, "mask": mask_tensor, "id": img_id}

