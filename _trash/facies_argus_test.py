import os
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import trange
from argparse import ArgumentParser, Namespace
from time import time

import torch

torch.backends.cudnn.benchmark = True
import torch.optim as optim

import segmentation_models_pytorch as smp
import argus
from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile, ReduceLROnPlateau
from pytorch_toolbelt.inference.tiles import ImageSlicer, CudaTileMerger
from _torch.losses import LovaszBCELoss
from _torch.data import get_data_loaders, SeismicFaciesDataset, DataLoader
import utils as U

seed = 0
np.random.seed(seed)
CLASS_NAMES = ["Basement", "SlopeMudA", "Deposit", "SlopeMudB", "SlopeValley", "Canyon"]
num_classes = len(CLASS_NAMES)


class SeismicFaciesModel(argus.Model):
    nn_module = smp.Unet
    optimizer = optim.SGD
    loss = LovaszBCELoss


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False,
                        default="argus_normalized")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--sub", type=str, required=True)
    parser.add_argument("--gpu", type=int, nargs='+', required=False)
    parser.add_argument("--workers", type=int, required=False, default=4)
    parser.add_argument("--batch", type=int, required=False, default=64)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    # load run folder and parameters
    model_path = os.path.join('./results', args.model)  # argus_5folds_0/fold_0/model.pth
    run_path = os.path.splitext(model_path)[0].split('/')[2]
    
    with open(os.path.join('./results', run_path, 'args.txt'), 'r') as fp:
        run = Namespace()
        run.__dict__.update(json.load(fp))
    
    patch_size = tuple(run.patch)
    
    test_img = np.load(os.path.join('/nas/home/fpicetti/datasets/seismic_facies/', run.dataset, 'data_test_1.npz'),
                       allow_pickle=True, mmap_mode='r')['data']
    train_img_shape = np.load(os.path.join('/nas/home/fpicetti/datasets/seismic_facies/', run.dataset, 'data_train.npz'),
                              allow_pickle=True, mmap_mode='r')['data'].shape
    label_dtype = np.uint8
    
    model = argus.load_model(model_path)
    
    tiler = ImageSlicer(train_img_shape[:-1] + (1,), tile_size=patch_size, tile_step=(1, 8))
    merger = CudaTileMerger(tiler.target_shape, 6, tiler.weight)
    
    test_labels = []
    test_img_T = test_img.transpose(2, 0, 1)
    
    for img_idx in trange(test_img_T.shape[0], desc='XZ section'):
        
        img = test_img_T[img_idx]
        
        tiles = [tile for tile in tiler.split(img[:, :, None])]
        
        for tiles_batch, coords_batch in DataLoader(list(zip(tiles, tiler.crops)), batch_size=args.batch):
            tiles_batch = tiles_batch.permute(0, 3, 1, 2)
            pred_batch = torch.softmax(model.predict(tiles_batch), axis=1).to(merger.image.device)
            
            merger.integrate_batch(pred_batch, coords_batch)
        
        merged_mask = merger.merge()
        merged_mask = merged_mask.permute(1, 2, 0).cpu().numpy()
        merged_mask = tiler.crop_to_orignal_size(merged_mask).argmax(2)
        
        test_labels.append(merged_mask)
    
    del test_img_T
    test_labels = np.stack(test_labels).transpose(1, 2, 0)
    
    outfile = os.path.join('./facies_aicrowd_submissions', '%s.npz' % args.sub)
    np.savez_compressed(outfile, prediction=test_labels.astype(label_dtype) + 1)
    print('Prediction saved to %s' % outfile)
    