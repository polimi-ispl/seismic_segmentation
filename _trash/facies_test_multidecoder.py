import os
import numpy as np
from time import time
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,\
    TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from _keras import architectures as a, data as d
from _keras.losses import sum_weighted_losses
from tensorflow.keras.models import load_model
import utils as u
import albumentations as albu
from argparse import ArgumentParser
import warnings

seed = 0
np.random.seed(seed)
warnings.filterwarnings("ignore")
CLASS_NAMES = ["Basement", "SlopeMudA", "Deposit", "SlopeMudB", "SlopeValley", "Canyon"]
num_classes = len(CLASS_NAMES)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Build a tensorboard logging page')
    parser.add_argument("--dataset", type=str, required=False,
                        default="train4.csv")
    parser.add_argument("--gpu", type=int, required=False, default=0)
    parser.add_argument("--workers", type=int, required=False, default=4)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument('--outpath', type=str, required=False, default='debug',
                        help='Run name in /../results/')
    # Training strategies
    parser.add_argument('--backbone', type=str, required=False, default='xception',
                        choices=['xception', 'multires'],
                        help='Name of the network backbone')
    parser.add_argument('--nf', type=int, required=False, default=16,
                        help='Numbers of generator channels')
    parser.add_argument("--separate_classes", action="store_true", default=False)
    parser.add_argument("--use_edges", action="store_true", default=False)
    parser.add_argument("--use_inputs", action="store_true", default=False)
    parser.add_argument("--batch", type=int, required=False, default=32)
    parser.add_argument("--shuffle", action="store_true", default=False)
    # Preprocessing and augmentation
    parser.add_argument("--hflip", action="store_true", default=False)
    parser.add_argument("--downscale", action="store_true", default=False)
    
    # Optimizer and Learning Rate Scheduler
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help='Learning Rate for optimizer')
    parser.add_argument('--lr_patience', type=int, default=10, required=False,
                        help='Number of iterations for decaying the Learning Rate')
    parser.add_argument('--lr_plateau_factor', type=float, default=.2, required=False,
                        help='Factor for reducing the Learning Rate on plateau')
    parser.add_argument('--loss_patience', type=int, default=10, required=False,
                        help='Number of iterations for stopping if the validation loss does not decrease.')
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to be used')
    # Segmentation Loss functions
    parser.add_argument('--loss_seg', nargs='+', type=str, required=False,
                        choices=['l2dist', 'l1dist', 'l2norm', 'l1norm', 'tv', 'bce', 'cce',
                                 'dice', 'hinge', 'chinge', 'focal', 'lovasz', 'clovasz'],
                        help='Segmentation loss names')
    parser.add_argument('--loss_seg_weights', nargs='+', type=float, required=False,
                        help='Segmentation loss weights')
    # Edging Loss functions
    parser.add_argument('--loss_edges', nargs='+', type=str, required=False,
                        choices=['l2dist', 'l1dist', 'mae', 'mse', 'l2norm', 'l1norm', 'tv'],
                        help='Edging loss names')
    parser.add_argument('--loss_edges_weights', nargs='+', type=float, required=False,
                        help='Edging loss weights')
    # Reconstruction Loss functions
    parser.add_argument('--loss_data', nargs='+', type=str, required=False,
                        choices=['l2dist', 'mse', 'mae', 'l1dist', 'l2norm', 'l1norm', 'tv'],
                        help='Data autoencoding loss names')
    parser.add_argument('--loss_data_weights', nargs='+', type=float, required=False,
                        help='Data autoencoding loss weights')
    args = parser.parse_args()
    
    if args.loss_seg is None:
        args.loss_seg = ["bce"]
    if len(args.loss_seg) == 1:
        args.loss_seg_weights = [1.]
    if args.separate_classes:
        if "cce" in [s.lower() for s in args.loss_seg]:
            print(
                "Warning: Categorical Crossentropy does not work with separate classes, switching to Binary Crossentropy...")
            args.loss_seg = ["bce" if s.lower() == "cce" else s.lower() for s in args.loss_seg]
        if len(args.loss_seg) == 1:
            args.loss_seg *= num_classes
        if len(args.loss_seg_weights) == 1:
            args.loss_seg_weights *= num_classes
    assert len(args.loss_seg) == len(args.loss_seg_weights), 'Segmentation Loss functions and weights mismatch'
    
    if not args.use_edges:
        del args.loss_edges, args.loss_edges_weights
    else:
        if args.loss_edges is None:
            args.loss_edges = ["mae"]
        if len(args.loss_edges) == 1:
            args.loss_edges_weights = [1.]
        assert len(args.loss_edges) == len(args.loss_edges_weights), 'Edging Loss functions and weights mismatch'
    
    if not args.use_inputs:
        del args.loss_data, args.loss_data_weights
    else:
        if args.loss_data is None:
            args.loss_data = ["mse"]
        if len(args.loss_data) == 1:
            args.loss_data_weights = [1.]
        assert len(args.loss_data) == len(args.loss_data), 'Autoencoding Loss functions and weights mismatch'
    
    return args


if __name__ == "__main__":
    args = parse_arguments()
    
    # dataset
    data_root = "/nas/home/fpicetti/datasets/seismic_facies/"
    dataset = os.path.join(data_root, args.dataset)
    
    # create output folder
    outpath = os.path.join('./results/', args.outpath)
    os.makedirs(outpath, exist_ok=True)
    
    with open(os.path.join(outpath, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    
    u.set_gpu(args.gpu)
    
    # Data generators
    aug = []
    if args.hflip:
        aug.append(albu.HorizontalFlip())
    if args.downscale:
        aug.append(albu.Downscale(scale_min=0.5, scale_max=0.95, p=0.1))
    
    datagen_train = d.CSVLoader(
        csv_file=dataset,
        phase="train",
        batch_size=args.batch,
        separate_classes=args.separate_classes,
        use_edges=args.use_edges,
        use_inputs=args.use_inputs,
        shuffle=args.shuffle,
        augmentation=albu.Compose(aug),
        preprocessing=None,
    )
    
    datagen_val = d.CSVLoader(
        csv_file=dataset,
        phase="val",
        batch_size=args.batch,
        separate_classes=args.separate_classes,
        use_edges=args.use_edges,
        use_inputs=args.use_inputs,
        shuffle=args.shuffle,
        augmentation=albu.Compose(aug),
        preprocessing=None,
    )
    
    patch_shape = datagen_train[0][0].shape[1:]
    # Build model
    K.clear_session()
    
    model = a.MultiDecoder(
        patch_shape,
        num_classes=num_classes,
        separate_classes=args.separate_classes,
        use_edges=args.use_edges,
        use_inputs=args.use_inputs,
        nf=args.nf, act="relu",
        backbone=args.backbone)
    
    model.save(os.path.join(outpath, '_model.h5'))
    
    if args.optimizer == 'adam':
        opt = Adam(lr=args.lr)
    elif args.optimizer == 'sgd':
        opt = SGD(lr=args.lr)
    elif args.optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=args.lr)
    
    # TODO fix loss_seg and loss_seg_weights lists for handling separate_classes case
    # if the user asks for a list of (weighted) losses, this is spread over all the segmentation outputs.
    # we should have a way to propagate the same loss function with different weights (for class weighting)
    if args.separate_classes:
        loss_segmentation = [lambda target, output: sum_weighted_losses(target[n], output[n],
                                                                        args.loss_seg[n], args.loss_seg_weights[n])
                             for n in range(num_classes)]
    else:  # the output is a vector of #num_classes elements; use all the loss_seg together
        loss_segmentation = [lambda target, output: sum_weighted_losses(target[0], output[0],
                                                                        args.loss_seg, args.loss_seg_weights)]
    
    _edges_idx = -2 if args.use_inputs else -1
    loss_edges = lambda target, output: sum_weighted_losses(target[_edges_idx], output[_edges_idx],
                                                            args.loss_edges, args.loss_edges_weights)
    
    loss_data = lambda target, output: sum_weighted_losses(target[-1], output[-1],
                                                           args.loss_data, args.loss_data_weights)
    
    losses = loss_segmentation
    if args.use_edges:
        losses.append(loss_edges)
    if args.use_inputs:
        losses.append(loss_data)
    
    model.compile(loss=losses, optimizer=opt)
    
    # checkpoint
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=args.lr_plateau_factor, patience=args.lr_patience,
                                   verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
                 EarlyStopping(monitor='val_loss', patience=args.loss_patience),
                 ModelCheckpoint(os.path.join(outpath, '_weights.h5'), monitor='val_loss',
                                 save_best_only=True, save_weights_only=True),
                 CSVLogger(os.path.join(outpath, 'training.log'), separator=",", append=False)]
    if args.tensorboard:
        callbacks.append(TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=args.batch,
                                     write_graph=True, write_grads=False, write_images=False,
                                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                     embeddings_data=None, update_freq='epoch'))
    # training
    start = time()
    train = model.fit(datagen_train, validation_data=datagen_val,
                      epochs=args.epochs, callbacks=callbacks, verbose=1)
    training_time = time() - start
    print('Finished training in %s' % u.sec2time(training_time))
    
    # save results
    mydict = {
        'server'    : u.machine_name(),
        'device'    : os.environ["CUDA_VISIBLE_DEVICES"],
        'train_time': u.sec2time(training_time),
        'history'   : train.history,
        'args'      : args
    }
    np.save(os.path.join(outpath, 'run.npy'), mydict)
    print('Saved to %s' % outpath)
