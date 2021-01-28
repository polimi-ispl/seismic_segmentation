import os
from collections import OrderedDict
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from argparse import Namespace
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from .architectures import MulResUnet3D, DiceLoss, Iou_pytorch, lovasz_hinge, init_weights, AttMulResUnet3D
seed = pl.seed_everything(42)


# Dataset
class SimuDataset(Dataset):
    """Dataset for simulated far image datas
    Args:
        csv_path (str): The path of the csv file saved the information of dataset
        phase (str): indicating the train, val, test.
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
            self,
            csv_path,
            phase,
            augmentation=None,
    ):
        
        df = pd.read_csv(csv_path, converters={'name': str})
        df_tr = df[df['mode'] == 'train']
        df_val = df[df['mode'] == 'val']
        df_test = df[df['mode'] == 'test']
        
        if phase == 'train':
            self.lbl_paths = df_tr['lbl_path'].values
            self.img_paths = df_tr['img_path'].values
        elif phase == 'val':
            self.lbl_paths = df_val['lbl_path'].values
            self.img_paths = df_val['img_path'].values
        else:
            self.lbl_paths = df_test['lbl_path'].values
            self.img_paths = df_test['img_path'].values
        
        self.augmentation = augmentation

    def __getitem__(self, i):

        # read data
        image = np.load(self.img_paths[i])[np.newaxis]
        mask = np.load(self.lbl_paths[i])[np.newaxis]
        image = (image - image.mean()) / image.std()

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return {'img': torch.from_numpy(image), 
                "lbl": torch.from_numpy(mask),
                'img_name': os.path.basename(self.img_paths[i]),
               'lbl_name':os.path.basename(self.lbl_paths[i])}

    def __len__(self):
        return len(self.img_paths)

def augmentation(image, mask):
    ind = np.random.randint(0, 4, size=1)[0]
    image = image.squeeze(0)
    mask = mask.squeeze(0)
    sample = {}
    sample['image'] = np.rot90(image, k=ind, axes=(0,1)).copy()[np.newaxis]
    sample['mask'] = np.rot90(mask, k=ind, axes=(0,1)).copy()[np.newaxis]
    return sample 

# PL Model
class UnetModel(pl.LightningModule):
    def __init__(self, hparams):
        super(UnetModel, self).__init__()

        self.hparams = hparams
        
        self.net = MulResUnet3D(1, 1)
        init_weights(self.net, hparams.init_type, hparams.init_gain)
#         self.net = smp.Unet(
#             encoder_name=hparams.ENCODER, 
#             encoder_weights=hparams.ENCODER_WEIGHTS, 
#             classes=1, 
#             activation=None)
    
#         checkpoint_dir = '/nas/home/fkong/data/farimage/simulation/fracture_new/preds/frac_multi/UNET/version_6/'
#         ckpt_name = [x for x in os.listdir(checkpoint_dir) if '.ckpt' in x][0]
#         chp = torch.load(os.path.join(checkpoint_dir, ckpt_name))

#         state_dict = OrderedDict([(k.replace('net.', ''), v) for k, v in chp['state_dict'].items()])
#         self.net.load_state_dict(state_dict)

        self.criterionDice = DiceLoss()  #   DiceLoss_iou()

        self.criterionCE = torch.nn.BCEWithLogitsLoss()

        self.criterionlozasz = lovasz_hinge()
        self.iou = Iou_pytorch()
        self.val_acc = 0
        # self.criterionweightBCE = weightcrossentropyloss()  
    def forward(self, x):
        # called with self(x)
        return self.net(x)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x = batch['img']
        y = batch['lbl']
        y_hat = self(x)
        loss_Dice = self.criterionDice(y_hat, y)
        loss_bce = self.criterionCE(y_hat, y)

        iou_metric = self.iou(y_hat, y)

        loss_all = loss_bce
        bar = {'iou': iou_metric}
        tensorboard_logs = {'train_bce': loss_bce, 'train_dice': 1 - loss_Dice, 'train_iou':iou_metric}
        return {'loss': loss_all, 'progress_bar':bar, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch['img']
        y = batch['lbl']
        y_hat = self(x)
        loss_Dice = 1 - self.criterionDice(y_hat, y)
        loss_bce = self.criterionCE(y_hat, y)
        iou_metric = self.iou(y_hat, y)

        tensorboard_logs = {'val_iou': iou_metric, 'val_dice': loss_Dice}
        return {'loss': loss_bce, 'iou': iou_metric, 'log':tensorboard_logs}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
#         avg_cara = torch.stack([x['cara'] for x in outputs]).mean()
        self.val_acc = avg_iou
        return {'val_loss': avg_loss, 'val_acc':avg_iou}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch['img']
        y = batch['lbl']
        y_hat= self(x)
        loss_Dice = 1 - self.criterionDice(y_hat, y)
        iou_metric = self.iou(y_hat, y)
        print('the iou of %s is %f' % (batch['img_name'], iou_metric.item()))
       
        bar = {'test_iou': iou_metric}
        logs = {'test_iou': iou_metric, 'test_dice':loss_Dice}
        return {'iou': iou_metric.item(), 'dice':loss_Dice.item(), 
                'progress_bar':bar, 'log':logs, 'gen':y_hat.sigmoid().cpu().numpy()}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        save_dir = os.path.join(self.hparams.results_dir, self.hparams.results_name, 'version_' + str(self.hparams.version))
        out_dict = {}
        for i, x in enumerate(outputs):
            out_dict['gen_%d' % i] = x['gen']
            out_dict['iou_%d' % i] = x['iou']
        
        avg_iou = np.array([x['iou'] for x in outputs]).mean()
        avg_dice = np.array([x['dice'] for x in outputs]).mean()
        out_dict['avg_iou'] = avg_iou
        out_dict['avg_dice'] = avg_dice
        np.save(os.path.join(save_dir, 'res.npy'), out_dict)
        return {'test_iou': avg_iou, 'test_dice':avg_dice}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.logger.log_hyperparams(self.hparams)
        
#         optimizer = torch.optim.SGD(self.net.parameters(), lr=self.hparams.lr, momentum=0.9)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr,
                                     betas=(self.hparams.b1, self.hparams.b2))
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
                        'interval': 'epoch'}
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        # REQUIRED
        train_dataset = SimuDataset(self.hparams.csv_path, 'train', 
                                    augmentation=False)
        return DataLoader(train_dataset, shuffle=True,
                          batch_size=self.hparams.batch_size, num_workers=int(self.hparams.num_threads)) 

    def val_dataloader(self):
        # OPTIONAL
        val_dataset = SimuDataset(self.hparams.csv_path, 'val')
        return DataLoader(val_dataset, shuffle=False,
                          batch_size=self.hparams.batch_size, num_workers=int(self.hparams.num_threads))

    def test_dataloader(self):
        # OPTIONAL
        test_data = SimuDataset(self.hparams.csv_path, 'test')
        return DataLoader(test_data, shuffle=False,
                          batch_size=1, num_workers=int(self.hparams.num_threads))
# Argmentation
if __name__ == '__main__':
    args = {
        'csv_path': '/nas/home/fkong/data/faultdetect/synthetic_wu/data.csv',
        'results_dir': '/nas/home/fkong/data/faultdetect/synthetic_wu/preds/',
        'results_name': 'multi3d',
        'batch_size': 4,
        'num_threads': 8,
        'lr': 0.001,
        'b1': 0.9,
        'b2': 0.999,
        'max_epochs': 200,
        'port': 8098,
        'display_freq': 100,
        'version':5,
        'init_type':'xavier',
        'init_gain':0.02
    }

    hyparams = Namespace(**args)

    segmodel = UnetModel(hyparams)

    earlystop = pl.callbacks.EarlyStopping('val_acc', mode='max', min_delta=1e-4, patience=10)
    tb_logger = pl.loggers.TensorBoardLogger(hyparams.results_dir, hyparams.results_name, version=hyparams.version)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(os.path.join(tb_logger.log_dir,
                                                    'version_' + str(tb_logger.version)), monitor='val_acc',
                                                    verbose=True)
    lr_logger = pl.callbacks.lr_logger.LearningRateLogger()
    cmd = 'tensorboard --logdir %s --port %s' % (tb_logger.log_dir, hyparams.port)
    print(cmd)

    gpus = [0, 1, 2, 3]

    trainer = pl.Trainer(early_stop_callback=earlystop, gpus=gpus, distributed_backend='dp', 
                        logger=tb_logger, max_epochs=hyparams.max_epochs, min_epochs=50,
                        callbacks=[lr_logger],
                        checkpoint_callback=checkpoint_callback)

    # checkpoint_dir = '/nas/home/fkong/data/farimage/simulation/fracture_new/preds/frac_multi/UNET/version_3/'
    # ckpt_name = [x for x in os.listdir(checkpoint_dir) if '.ckpt' in x][0]

    # segmodel = UnetModel.load_from_checkpoint(
    #     os.path.join(checkpoint_dir, ckpt_name),
    #     map_location=None)
    # segmodel.current_epoch = 0

    trainer.fit(segmodel)
    # trainer.test(segmodel)
    trainer = pl.Trainer(gpus=[0])

    checkpoint_dir = os.path.join(tb_logger.log_dir)
    ckpt_name = [x for x in os.listdir(checkpoint_dir) if '.ckpt' in x][0]

    segmodel = UnetModel.load_from_checkpoint(
        os.path.join(checkpoint_dir, ckpt_name),
        map_location=None)

    trainer.test(segmodel)