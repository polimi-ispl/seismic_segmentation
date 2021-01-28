from typing import Union
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel


def cast_device(device):
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, (list, tuple)):
        if len(device) == 1:
            return torch.device(device[0])
        elif len(device) == 0:
            raise ValueError("Empty list of devices")
        else:
            return [torch.device(d) for d in device]
    else:
        return torch.device(device)


def device_to_str(device):
    if isinstance(device, (list, tuple)):
        return [str(d) for d in device]
    else:
        return str(device)
    
    
def get_nn_module(nn_module: torch.nn.Module) -> torch.nn.Module:
    if isinstance(nn_module, (DataParallel, DistributedDataParallel)):
        return nn_module.module
    else:
        return nn_module
    
    
def set_device(nn_module, device: Union[str, torch.device], loss=None):
    device = cast_device(device)
    nn_module = get_nn_module(nn_module)
    
    if isinstance(device, (list, tuple)):
        device_ids = []
        for dev in device:
            if dev.type != 'cuda':
                raise ValueError("Non cuda device in list of devices")
            if dev.index is None:
                raise ValueError("Cuda device without index in list of devices")
            device_ids.append(dev.index)
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("Cuda device indices must be unique")
        nn_module = DataParallel(nn_module, device_ids=device_ids)
        device = device[0]
    
    nn_module = nn_module.to(device)
    if loss is not None:
        loss = loss.to(device)
        return nn_module, loss
    else:
        return nn_module
