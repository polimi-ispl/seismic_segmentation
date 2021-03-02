from typing import Union, List, Optional, Dict
from pathlib import Path
from collections import OrderedDict, defaultdict
from torch.utils.data import DataLoader
from .dataset import ArrayDataset
import numpy as np
from sklearn.model_selection import train_test_split
from catalyst.contrib.tools.tensorboard import SummaryReader


def get_loaders(images: np.ndarray, masks: np.ndarray,
                random_state: int = 42, valid_size: float = 0.2,
                batch_size: int = 32, num_workers: int = 4,
                split_shuffle: bool = False, epoch_shuffle: bool = True,
                train_transforms_fn=None, valid_transforms_fn=None) -> dict:
    indices = np.arange(images.shape[0])
    
    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(indices,
                                                    test_size=valid_size,
                                                    random_state=random_state,
                                                    shuffle=split_shuffle)
    
    # Creates our train dataset
    train_dataset = ArrayDataset(
        images=images[train_indices],
        masks=masks[train_indices],
        transforms=train_transforms_fn
    )
    print('train dataset built')
    
    # Creates our valid dataset
    valid_dataset = ArrayDataset(
        images=images[valid_indices],
        masks=masks[valid_indices],
        transforms=valid_transforms_fn
    )
    print('validation dataset built')
    
    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=epoch_shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    
    # And excpect to get an OrderedDict of loaders
    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
    print('loader created')
    return loaders


def get_tensorboard_log(logdir: Union[str, Path], step: Optional[str] = "epoch",
                        metrics: Optional[List[str]] = None) -> defaultdict:
    
    def _get_scalars(logdir: Union[str, Path], metrics: Optional[List[str]], step: str) -> Dict[str, List]:
        summary_reader = SummaryReader(logdir, types=["scalar"])
        
        items = defaultdict(list)
        for item in summary_reader:
            if step in item.tag and (
                    metrics is None or any(m in item.tag for m in metrics)
            ):
                items[item.tag].append(item)
        return items
    
    logdir = Path(logdir)
    
    logdirs = {x.name.replace("_log", ""): x for x in logdir.glob("**/*") if x.is_dir() and str(x).endswith("_log")}
    
    scalars_per_loader = {key: _get_scalars(inner_logdir, metrics, step) for key, inner_logdir in logdirs.items()}
    
    scalars_per_metric = defaultdict(dict)
    for key, value in scalars_per_loader.items():
        for key2, value2 in value.items():
            scalars_per_metric[key2][key] = value2
    
    return scalars_per_metric
