import os
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from argparse import ArgumentParser
from time import time

import torch

torch.backends.cudnn.benchmark = True
from torch.optim import SGD, Adam

import albumentations as A
import segmentation_models_pytorch as smp
import argus
from argus.callbacks import MonitorCheckpoint, EarlyStopping, LoggingToFile, ReduceLROnPlateau
from _torch.losses import LovaszBCELoss
from _torch.data import get_data_loaders, SeismicFaciesDataset
import utils as U

seed = 0
np.random.seed(seed)
CLASS_NAMES = ["Basement", "SlopeMudA", "Deposit", "SlopeMudB", "SlopeValley", "Canyon"]
num_classes = len(CLASS_NAMES)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False,
                        default="argus_normalized")
    parser.add_argument("--gpu", type=int, nargs='+', required=False)
    parser.add_argument("--workers", type=int, required=False, default=16)
    parser.add_argument('--outpath', type=str, required=False, default='debug',
                        help='Run name in ./results/')
    parser.add_argument('--max_saves', type=int, required=False)
    
    # Training strategies
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument('--encoder', type=str, required=False, default='efficientnet-b3',
                        help='https://github.com/qubvel/segmentation_models.pytorch#encoders')
    parser.add_argument('--decoder', type=str, required=False, default='scse')
    parser.add_argument('--num_folds', type=int, required=False, default=5,
                        help='Numbers of folds')
    parser.add_argument("--batch", type=int, required=False, default=32)
    
    # Preprocessing and augmentation
    parser.add_argument('--patch', required=False, type=int, nargs='+')
    parser.add_argument("--aug", default=True, type=bool, required=False,
                        help="Apply data augmentation")
    
    # Optimizer and Learning Rate Scheduler
    parser.add_argument('--opt', type=str, default='sgd', required=False,
                        choices=['sgd', 'adam'], help='Optimizer name')
    parser.add_argument('--lr', type=float, default=1e-2, required=False,
                        help='Learning Rate for optimizer')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, required=False)
    parser.add_argument('--weight_decay', type=float, default=0.0001, required=False)
    parser.add_argument('--dampening', type=float, required=False, default=0.)
    parser.add_argument('--nesterov', type=bool, required=False, default=False)
    parser.add_argument('--eps', type=float, required=False, default=1e-08)
    parser.add_argument('--betas', type=float, nargs='+', required=False)
    parser.add_argument('--amsgrad', type=bool, required=False, default=False)
    parser.add_argument('--lr_patience', type=int, default=5, required=False,
                        help='Number of iterations for decaying the Learning Rate')
    parser.add_argument('--lr_plateau_factor', type=float, default=.64, required=False,
                        help='Factor for reducing the Learning Rate on plateau')
    parser.add_argument('--loss_patience', type=int, default=20, required=False,
                        help='Number of iterations for stopping if the validation loss does not decrease.')
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to be used')
    
    # Segmentation Loss functions
    parser.add_argument('--lovasz', type=float, required=False, default=0.75,
                        help='Segmentation Lovasz weight')
    parser.add_argument('--bce', type=float, required=False, default=0.25,
                        help='Segmentation BCE weight')
    args = parser.parse_args()
    
    if args.gpu is None:
        args.gpu = [0]
    
    if args.patch is None:
        args.patch = [896, 256]
    
    if args.opt == 'sgd':
        # del args.betas, args.eps, args.amsgrad
        args.opt_param = dict(
            lr=args.lr,
            momentum=args.sgd_momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)
    else:
        # del args.sgd_momentum, args.nesterov, args.dampening
        args.opt_param = dict(
            lr=args.lr,
            betas=(0.9, 0.999) if args.betas is None else args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    # create output folder
    outpath = os.path.join('./results/', args.outpath)
    os.makedirs(outpath, exist_ok=True)
    
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    
    patch_size = tuple(args.patch)
    
    params = {
        'nn_module': {
            'encoder_name'          : args.encoder,
            'decoder_attention_type': args.decoder,
            'classes'               : num_classes,
            'in_channels'           : 1,
            'activation'            : None
        },
        'loss'     : {
            'lovasz_weight': args.lovasz,
            'ce_weight'    : args.bce,
        },
        'optimizer': args.opt_param,
        'device'   : [f'cuda:{gpu}' for gpu in args.gpu]
    }
    
    
    class SeismicFaciesModel(argus.Model):
        nn_module = smp.Unet
        optimizer = SGD if args.opt == 'sgd' else Adam
        loss = LovaszBCELoss

    model = SeismicFaciesModel(params)

    # load data
    dataset = os.path.join('/nas/home/fpicetti/datasets/seismic_facies/', args.dataset)
    
    train_img = np.load(os.path.join(dataset, 'data_train.npz'),
                        allow_pickle=True, mmap_mode='r')['data']
    train_labels = np.load(os.path.join(dataset, 'labels_train.npz'),
                           allow_pickle=True, mmap_mode='r')['labels']
    
    # Data generators
    if args.aug:
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Downscale(scale_min=0.5, scale_max=0.95, p=0.1),
            A.RandomCrop(p=1, height=patch_size[0], width=patch_size[1]),
        ])
    
    dataset = SeismicFaciesDataset(
        img=train_img,
        labels=train_labels,
        aug=aug if args.aug else None)
    
    # create folds
    cols = [0 for _ in range(train_img.shape[1])] + [1 for _ in range(train_img.shape[2])]
    if args.num_folds > 1:
        folds = list(StratifiedKFold(n_splits=args.num_folds, random_state=42, shuffle=True).split(X=cols, y=cols))
    else:
        folds = [train_test_split(cols, test_size=.1, random_state=42, shuffle=False)]
    
    # training
    start = time()
    for i, (train_index, test_index) in enumerate(folds):
        # model = SeismicFaciesModel(params)
        
        train_loader, val_loader = get_data_loaders(
            dataset,
            batch_size=args.batch,
            train_index=train_index,
            test_index=test_index,
            num_workers=args.workers,
            pad_min=(1024, 800))
        
        callbacks = [
            MonitorCheckpoint(dir_path=os.path.join(outpath, f'fold_{i}'),
                              monitor='val_loss', max_saves=args.max_saves,
                              file_format='model.pth'),
            ReduceLROnPlateau(monitor='val_loss', patience=args.lr_patience,
                              factor=args.lr_plateau_factor, min_lr=1e-8),
            EarlyStopping(monitor='val_loss', patience=args.loss_patience),
            LoggingToFile(os.path.join(outpath, f'fold_{i}.log')),
        ]
        
        model.fit(
            train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            metrics=['loss'],
            callbacks=callbacks,
            metrics_on_train=False)
    training_time = time() - start
    print('Finished training in %s' % U.sec2time(training_time))
