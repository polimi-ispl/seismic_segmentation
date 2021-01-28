import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union
from collections import defaultdict
from pathlib import Path
from catalyst.contrib.tools.tensorboard import SummaryReader
from catalyst.callbacks import BatchMetricCallback
from catalyst.utils.torch import get_activation_fn
from utils import to_categorical, binary_mask


class I2IDataset(Dataset):
    """Build a Image to Image dataset"""
    
    def __init__(self, images: np.ndarray, masks: np.ndarray = None, transforms=None) -> None:
        self.images = images
        self.transforms = transforms

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> dict:
    
        result = {"image": self.images[idx]}
    
        if self.transforms is not None:
            result = self.transforms(**result)
        
        result["mask"] = result["image"]
        
        return result

        
class SegmentationDataset(Dataset):
    
    def __init__(self, images: np.ndarray, masks: np.ndarray = None, transforms=None) -> None:
        # arrays of patches
        
        self.images = images
        self.masks = masks
        
        self.transforms = transforms
    
    def __len__(self) -> int:
        return self.images.shape[0]
    
    def __getitem__(self, idx: int) -> dict:
        
        result = {'image': self.images[idx]}
        
        if self.masks is not None:
            # note: the squeeze() is necessary for the dimensions to be the same
            result['mask'] = self.masks[idx].squeeze()
            
        if self.transforms is not None:
            result = self.transforms(**result)
            
        return result


def get_loaders(images: np.ndarray, masks: np.ndarray, dataset: Dataset,
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
    train_dataset = dataset(
        images=images[train_indices],
        masks=masks[train_indices],
        transforms=train_transforms_fn
    )
    
    # Creates our valid dataset
    valid_dataset = dataset(
        images=images[valid_indices],
        masks=masks[valid_indices],
        transforms=valid_transforms_fn
    )
    
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
        drop_last=True,
    )
    
    # And excpect to get an OrderedDict of loaders
    loaders = OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader
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


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    weights: list = None,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Softmax2d",
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        beta (float): beta param for f_score
        weights (torch.Tensor): A list of classes weights
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        float: F_1 score
    """
    activation_fn = get_activation_fn(activation)
    
    outputs = activation_fn(outputs)
    
    if threshold is not None:
        outputs = (outputs > threshold).float()
    
    num_classes = int(outputs.shape[1])
    
    if targets.shape != outputs.shape:
        targets = to_categorical(targets)
    
    def _f1_single(targets, outputs, beta=beta, eps=eps):
        
        tp = torch.sum(targets * outputs)
        fp = torch.sum(outputs) - tp
        fn = torch.sum(targets) - tp
        
        precision_plus_recall = ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
        
        return ((1 + beta ** 2) * tp + eps) / precision_plus_recall
    
    if weights is None:
        weights = [1 / num_classes] * num_classes
    
    return sum([weights[i] * _f1_single(targets[:, i], outputs[:, i]) for i in range(num_classes)])


class F1ScoreMulticlass(BatchMetricCallback):
    """F1 score metric callback."""
    
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "f1",
            weights: list = None,
            beta: float = 1.0,
            eps: float = 1e-7,
            threshold: float = None,
            activation: str = "Softmax2d",
    ):
        """
        Args:
            input_key: input key to use for iou calculation
                specifies our ``y_true``
            output_key: output key to use for iou calculation;
                specifies our ``y_pred``
            prefix: key to store in logs
            weights: list of class weights
            beta: beta param for f_score
            eps: epsilon to avoid zero division
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax2d'``
        """
        super().__init__(
            prefix=prefix,
            metric_fn=f1_score,
            input_key=input_key,
            output_key=output_key,
            beta=beta,
            weights=weights,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


def f1_single(targets, outputs, eps=1e-10):
    tp = torch.sum(targets * outputs)
    fp = torch.sum((1-targets) * outputs)
    fn = torch.sum(targets * (1-outputs))
    
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    
    return 1 - torch.mean(f1)
