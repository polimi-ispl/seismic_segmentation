from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from copy import copy
import albumentations as A
import utils as u


class SeismicFaciesDataset(Dataset):
    def __init__(self, img, labels=None, horizon=False, axis='xy', aug=None, preproc=None, channel_first=True,
                 horizon_yield='channel', depth=False):
        self.img = img
        self.labels = labels
        self.xaxis = self.img.shape[1]
        self.yaxis = self.img.shape[2]
        self.axis = axis
        self.aug = aug
        self.preproc = preproc
        self.channel_first = channel_first
        self.horizon = horizon
        if horizon_yield == 'channel':
            self.horizon_yield = 'channel'
        elif horizon_yield == 'input':
            self.horizon_yield = 'input'
        else:
            raise ValueError('horizon_yield has to be either channel or input')
        self.depth = depth
    
    def __len__(self):
        len = 0
        if 'x' in self.axis:
            len += self.xaxis
        if 'y' in self.axis:
            len += self.yaxis
        return len
    
    def __getitem__(self, idx):
        if self.axis == 'x':
            image = self.img[:, idx]
            mask = self.labels[:, idx] if self.labels is not None else None
        elif self.axis == 'y':
            image = self.img[:, :, idx]
            mask = self.labels[:, :, idx] if self.labels is not None else None
        else:
            if idx < self.xaxis:
                image = self.img[:, idx]
                mask = self.labels[:, idx] if self.labels is not None else None
            else:
                image = self.img[:, :, idx - self.xaxis]
                mask = self.labels[:, :, idx - self.xaxis] if self.labels is not None else None
        
        image = np.expand_dims(image, -1)
        # mask = np.expand_dims(mask, -1)
        
        if self.preproc is not None:
            preproc = self.preproc(image=image, mask=mask)
            image, mask = preproc['image'], preproc['mask']
        
        if self.aug is not None:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1))
        
        if self.labels is not None:
            if self.horizon:
                hor = u.compute_edges(np.expand_dims(mask, (0, -1))).squeeze()
                if self.horizon_yield == 'channel':
                    mask = np.concatenate((mask[None, :, :], hor[None, :, :]), axis=0)
                    return image, mask
                else:
                    return image, mask, hor
            else:
                return image, mask
        else:
            return image


def get_data_loaders(dataset, batch_size, train_index, test_index, num_workers=16, pad_min=(1024, 800)):
    train_dataset, test_dataset = Subset(dataset, train_index), Subset(copy(dataset), test_index)
    test_dataset.dataset.aug = A.PadIfNeeded(p=1, min_height=pad_min[0], min_width=pad_min[1])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader
