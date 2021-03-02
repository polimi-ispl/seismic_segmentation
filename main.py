from typing import Callable, List, Tuple
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import trange
from args import parse_arguments
import torch
import catalyst

# models
from catalyst.dl import SupervisedRunner
import segmentation_models_pytorch as smp

# optimizer
from catalyst.contrib.nn import RAdam, Lookahead

# losses
from catalyst.contrib.nn import DiceLoss
from catalyst.contrib.nn.criterion.lovasz import LovaszLossBinary

# metrics
from catalyst.dl import IouCallback, CriterionCallback, MetricAggregationCallback, EarlyStoppingCallback
from catalyst.contrib.callbacks import DrawMasksCallback

# utilities
from pytorch_toolbelt.utils import count_parameters
import utils as u
import ttach

save_opts = {'format':'pdf', 'dpi':150, 'bbox_inches':'tight'}

SEED = 42
catalyst.utils.set_global_seed(SEED)
catalyst.utils.prepare_cudnn(deterministic=True)


def main():
    args = parse_arguments()

    
if __name__ == "__main__":
    main()
    