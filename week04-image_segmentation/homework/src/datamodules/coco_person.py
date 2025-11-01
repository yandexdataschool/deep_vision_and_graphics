from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import PersonSegmentationDataset


class CocoPersonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_ann: str,
        val_ann: str,
        test_ann: Optional[str] = None,
        image_size: Tuple[int, int] = (360, 480),
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        depth_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir).resolve()
        self.train_ann = Path(train_ann).resolve()
        self.val_ann = Path(val_ann).resolve()
        self.test_ann = Path(test_ann).resolve() if test_ann else None
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.depth_dir = Path(depth_dir).resolve() if depth_dir else None

        self._train_dataset: Optional[PersonSegmentationDataset] = None
        self._val_dataset: Optional[PersonSegmentationDataset] = None
        self._test_dataset: Optional[PersonSegmentationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            if self._train_dataset is None:
                self._train_dataset = PersonSegmentationDataset(
                    data_dir=self.data_dir,
                    ann_file=self.train_ann,
                    image_size=self.image_size,
                    augment=True,
                    depth_dir=self.depth_dir,
                )

            if self._val_dataset is None:
                self._val_dataset = PersonSegmentationDataset(
                    data_dir=self.data_dir,
                    ann_file=self.val_ann,
                    image_size=self.image_size,
                    augment=False,
                    depth_dir=self.depth_dir,
                )

        if stage in (None, "test") and self.test_ann is not None and self._test_dataset is None:
            self._test_dataset = PersonSegmentationDataset(
                data_dir=self.data_dir,
                ann_file=self.test_ann,
                image_size=self.image_size,
                augment=False,
                depth_dir=self.depth_dir,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self._test_dataset is None:
            return None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


