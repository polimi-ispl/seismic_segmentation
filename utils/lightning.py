import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .dataset import ArrayDataset


__all__ = [
    "ArrayDataModule",
]


class ArrayDataModule(pl.LightningDataModule):
    
    def __init__(self, images: np.ndarray, masks: np.ndarray = None,
                 valid_size: float = 0.2, seed=42,
                 train_transforms=None, valid_transforms=None, batch_size: int = 32,
                 pin_memory=False, drop_last=False, split_shuffle=True, epoch_shuffle=True,
                 num_workers=4):
        super(ArrayDataModule, self).__init__()
        
        assert images.shape[0] == masks.shape[0]
        
        self.batch_size = batch_size
        self.images = images
        self.masks = masks
        self.transforms_train = train_transforms
        self.transforms_valid = valid_transforms
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.split_shuffle = split_shuffle
        self.epoch_shuffle = epoch_shuffle
        self.valid_size = valid_size
        self.num_workers = num_workers
        
        self.train_idx, self.valid_idx = train_test_split(np.arange(self.images.shape[0]),
                                                          test_size=self.valid_size,
                                                          random_state=seed,
                                                          shuffle=self.split_shuffle)
    
    def train_dataloader(self):
        loader = DataLoader(
            ArrayDataset(self.images[self.train_idx], self.masks[self.train_idx], self.transforms_train),
            batch_size=self.batch_size,
            shuffle=self.epoch_shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(
            ArrayDataset(self.images[self.valid_idx], self.masks[self.valid_idx], self.transforms_valid),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader
    
    def test_dataloader(self, test_images, test_masks):
        loader = DataLoader(
            ArrayDataset(test_images, test_masks, self.transforms_valid),
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=self.pin_memory
        )
        return loader
