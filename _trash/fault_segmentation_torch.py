import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import numpy as np
from _torch.architectures import MulResUnet3D, DiceLoss, Iou_pytorch, lovasz_hinge, init_weights, AttMulResUnet3D
from _torch.data import CSVDataset
import utils as U
import albumentations as albu

seed = pl.seed_everything(42)


def albu_augmentation(sample):
    transforms = albu.Compose([
        albu.HorizontalFlip(),
        albu.Lambda(image=U.remove_mean_normalize_std, mask=None)
    ])
    return transforms(**sample)


def augmentation(image, mask):
    ind = np.random.randint(0, 4, size=1)[0]
    image = image.squeeze(0)
    mask = mask.squeeze(0)
    sample = {}
    sample['image'] = np.rot90(image, k=ind, axes=(0, 1)).copy()[np.newaxis]
    sample['mask'] = np.rot90(mask, k=ind, axes=(0, 1)).copy()[np.newaxis]
    return sample


# PL Model
class UnetModel(pl.LightningModule):
    def __init__(self, args, ch_in=1, ch_out=1):
        super(UnetModel, self).__init__()
        
        self.args = args
        
        self.net = MulResUnet3D(nc_in=ch_in, nc_out=ch_out)
        init_weights(self.net, args.init_type, args.init_gain)
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
        
        self.criterionDice = DiceLoss()  # DiceLoss_iou()
        
        self.criterionCE = torch.nn.BCEWithLogitsLoss()
        
        self.criterionLovasz = lovasz_hinge()
        self.iou = Iou_pytorch()
        self.val_acc = 0
        # self.criterionweightBCE = weightcrossentropyloss()  
    
    def forward(self, x):
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
        tensorboard_logs = {'train_bce': loss_bce, 'train_dice': 1 - loss_Dice, 'train_iou': iou_metric}
        return {'loss': loss_all, 'progress_bar': bar, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch['img']
        y = batch['lbl']
        y_hat = self(x)
        loss_Dice = 1 - self.criterionDice(y_hat, y)
        loss_bce = self.criterionCE(y_hat, y)
        iou_metric = self.iou(y_hat, y)
        
        tensorboard_logs = {'val_iou': iou_metric, 'val_dice': loss_Dice}
        return {'loss': loss_bce, 'iou': iou_metric, 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        #         avg_cara = torch.stack([x['cara'] for x in outputs]).mean()
        self.val_acc = avg_iou
        return {'val_loss': avg_loss, 'val_acc': avg_iou}
    
    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x = batch['img']
        y = batch['lbl']
        y_hat = self(x)
        loss_Dice = 1 - self.criterionDice(y_hat, y)
        iou_metric = self.iou(y_hat, y)
        print('the iou of %s is %f' % (batch['img_name'], iou_metric.item()))
        
        bar = {'test_iou': iou_metric}
        logs = {'test_iou': iou_metric, 'test_dice': loss_Dice}
        return {'iou'         : iou_metric.item(), 'dice': loss_Dice.item(),
                'progress_bar': bar, 'log': logs, 'gen': y_hat.sigmoid().cpu().numpy()}
    
    def test_epoch_end(self, outputs):
        # OPTIONAL
        save_dir = os.path.join(self.args.outpath, self.args.outname, 'version_%d' + self.args.version)
        out_dict = {}
        for i, x in enumerate(outputs):
            out_dict['gen_%d' % i] = x['gen']
            out_dict['iou_%d' % i] = x['iou']
        
        avg_iou = np.array([x['iou'] for x in outputs]).mean()
        avg_dice = np.array([x['dice'] for x in outputs]).mean()
        out_dict['avg_iou'] = avg_iou
        out_dict['avg_dice'] = avg_dice
        np.save(os.path.join(save_dir, 'res.npy'), out_dict)
        return {'test_iou': avg_iou, 'test_dice': avg_dice}
    
    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        self.logger.log_hyperparams(self.args)
        
        if self.args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr,
                                         betas=(self.args.b1, self.args.b2))
        
        lr_scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
                        'interval' : 'epoch'}
        
        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        # REQUIRED
        train_dataset = CSVDataset(self.args.csv_path, 'train', augmentation=False)
        return DataLoader(train_dataset, shuffle=True, batch_size=self.args.batch_size,
                          num_workers=int(self.args.num_threads))
    
    def val_dataloader(self):
        # OPTIONAL
        val_dataset = CSVDataset(self.args.csv_path, 'val')
        return DataLoader(val_dataset, shuffle=False, batch_size=self.args.batch_size,
                          num_workers=int(self.args.num_threads))
    
    def test_dataloader(self):
        # OPTIONAL
        test_data = CSVDataset(self.args.csv_path, 'test')
        return DataLoader(test_data, shuffle=False, batch_size=1,
                          num_workers=int(self.args.num_threads))


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=False,
                        default='/nas/home/fkong/data/faultdetect/synthetic_wu/data.csv')
    parser.add_argument("--gpus", nargs='+', type=int, required=False, default=[0])
    
    parser.add_argument("--outpath", type=str, required=False, default='debug')
    parser.add_argument("--outname", type=str, required=False, default='debug0')
    parser.add_argument("--version", type=int, required=False, default=5)

    parser.add_argument("--batch_size", type=int, required=False, default=4)
    parser.add_argument("--num_threads", type=int, required=False, default=8)
    parser.add_argument("--max_epochs", type=int, required=False, default=200)
    parser.add_argument("--port", type=int, required=False, default=8098)
    parser.add_argument("--display_freq", type=int, required=False, default=100)
    parser.add_argument("--init_type", type=str, required=False, default='xavier',
                        choices=['xavier', 'normal', 'kaiming', 'orthogonal'])
    parser.add_argument("--init_gain", type=float, required=False, default=.02)
    parser.add_argument("--optimizer", type=str, required=False, default='adam')
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--b1", type=float, required=False, default=.9)
    parser.add_argument("--b2", type=float, required=False, default=.999)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    segmodel = UnetModel(args)
    
    earlystop = pl.callbacks.EarlyStopping('val_acc', mode='max', min_delta=1e-4, patience=10)
    tb_logger = pl.loggers.TensorBoardLogger(args.outpath, args.outname, version=args.version)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(os.path.join(tb_logger.log_dir, 'version_%d' % tb_logger.version),
                                                       monitor='val_acc', verbose=True)
    lr_logger = pl.callbacks.lr_logger.LearningRateLogger()
    cmd = 'tensorboard --logdir %s --port %s' % (tb_logger.log_dir, args.port)
    print(cmd)
    
    trainer = pl.Trainer(early_stop_callback=earlystop, gpus=args.gpus, distributed_backend='dp',
                         logger=tb_logger, max_epochs=args.max_epochs, min_epochs=100,
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
