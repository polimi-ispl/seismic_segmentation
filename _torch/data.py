import os
from copy import copy
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset, Subset
import albumentations as A
import utils as u


class CSVDataset(Dataset):
    """Dataset object based on a CSV file
    Args:
        csv_path (str): The path of the csv file containing the dataset information
        phase (str): indicating the train, val, test.
        augmentation (albumentations.Compose): sample transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): sample preprocessing pipeline
            (e.g. normalization, shape manipulation, etc.)
    """
    
    def __init__(self, csv_path, phase, augmentation=None, preprocessing=None):
        
        df = pd.read_csv(csv_path, converters={'name': str})
        df_tr = df[df['mode'] == 'train']
        df_val = df[df['mode'] == 'val']
        df_test = df[df['mode'] == 'test']
        
        if phase == 'train':
            self.mask_paths = df_tr['lbl_path'].values
            self.image_paths = df_tr['img_path'].values
        elif phase == 'val':
            self.mask_paths = df_val['lbl_path'].values
            self.image_paths = df_val['img_path'].values
        else:
            self.mask_paths = df_test['lbl_path'].values
            self.image_paths = df_test['img_path'].values
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        image = np.load(self.image_paths[i])[np.newaxis]
        mask = np.load(self.mask_paths[i])[np.newaxis]
        
        # apply preprocessing
        if self.preprocessing is not None:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return {'img'     : torch.from_numpy(image),
                "lbl"     : torch.from_numpy(mask),
                'img_name': os.path.basename(self.image_paths[i]),
                'lbl_name': os.path.basename(self.mask_paths[i])}
    
    def __len__(self):
        return len(self.image_paths)


class DataGenerator(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        batch_x = self.x[index]
        batch_y = self.y[index]
        return np.expand_dims(batch_x, axis=0), batch_y


