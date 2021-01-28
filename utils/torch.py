import torch


def to_categorical(in_content, num_classes=None):
    if num_classes is None:
        num_classes = int(in_content.max()) + 1
    
    shape = in_content.shape[0], num_classes, *in_content.shape[2:]
    
    temp = torch.zeros(shape).transpose(0, 1)
    
    for i in range(num_classes):
        temp[i, (in_content == i).transpose(0, 1).squeeze(0)] = 1
    
    return temp.transpose(0, 1)


__all__ = [
    "to_categorical",
]
