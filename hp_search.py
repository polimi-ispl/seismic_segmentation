import argparse
import os

from catalyst.dl import AccuracyCallback
from catalyst.dl import SupervisedRunner
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import segmentation_models_pytorch as smp
import optuna
from optuna.integration import CatalystPruningCallback

smp.encoders

CLASSES = 10


class Net(nn.Module):
    def __init__(self, trial, classes=1, channels=1, act=None):
        super().__init__()
        
        net = trial.suggest_categorical("net", ["unet", "fpn", "psp", "linknet"])  # todo add PAN (check smp)
        backbone = trial.suggest_categorical("backbone", list(smp.encoders.encoders.keys()))
        depth = trial.suggest_int("depth", 2, 5)
        use_bn = trial.suggest_int("use_bn", 0, 1)
        dec_ch = trial.suggest_int
        if net == "unet":
            model = smp.Unet(
                classes=classes,
                in_channels=channels,
                encoder_name=backbone,
                activation=act,
                encoder_depth=depth,
                encoder_weights=None,
                decoder_use_batchnorm=bool(use_bn),
                decoder_channels=(256, 128, 64, 32, 16),
                decoder_attention_type=None,
            )
        elif net == 'linknet':
            model = smp.Linknet(
                classes=classes,
                in_channels=channels,
                encoder_name=backbone,
                activation=act,
                encoder_depth=depth,
                encoder_weights=None,
            )
        elif net == "fpn":
            model = smp.FPN(
                classes=classes,
                in_channels=channels,
                encoder_name=backbone,
                activation=act,
                encoder_weights=None,
                decoder_pyramid_channels=256,
                decoder_segmentation_channels=128,
                decoder_merge_policy="add",
                decoder_dropout=0.2,
                upsampling=4,
            )
        elif net == "psp":
            model = smp.PSPNet(
                classes=classes,
                in_channels=channels,
                encoder_name=backbone,
                activation=act,
                encoder_weights=None,
                encoder_depth=depth,
                psp_out_channels=512,
                psp_use_batchnorm=bool(use_bn),
                psp_dropout=0.2,
                upsampling=8,
            )
        elif net == "pan":
            model = smp.PAN(
                classes=classes,
                in_channels=channels,
                encoder_name=backbone,
                activation=act,
                encoder_weights=None,
                upsampling=4,
                decoder_channels=32,
                encoder_dilation=True,
            )
        self.layers = []
        self.dropouts = []
        
        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim
        
        self.layers.append(nn.Linear(input_dim, CLASSES))
        
        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)
        
        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)
    
    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


loaders = {
    "train": DataLoader(
        datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
        batch_size=100,
        shuffle=True,
    ),
    "valid": DataLoader(
        datasets.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
        batch_size=100,
    ),
}


def objective(trial):
    logdir = "./results/hp_search/"
    num_epochs = 50
    
    model = Net(trial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()
    
    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        callbacks=[
            AccuracyCallback(),
            CatalystPruningCallback(
                trial, metric="accuracy01"
            ),  # top-1 accuracy as metric for pruning
        ],
    )
    
    return runner.state.valid_metrics["accuracy01"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalyst example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()
    
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=10, timeout=600)
    
    print("Number of finished trials: {}".format(len(study.trials)))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: {}".format(trial.value))
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
