import albumentations as A


__all__ = [
    "pre_transform",
    "aug_transform",
    "tensor_transforms",
    "phase_transforms",
    "compose",
]


def pre_transform(target_size):
    return [A.Resize(target_size[0], target_size[1], p=1)]


def aug_transform(flip=0.5, downscale=0., elastic=0.):
    aug = [A.HorizontalFlip(p=flip),]
    if downscale > 0.:
        aug.append(A.Downscale(scale_min=0.5, scale_max=0.95, p=downscale))
    if elastic > 0.:
        aug.append(A.ElasticTransform(alpha=10, sigma=10, alpha_affine=0., p=elastic))
    
    return aug


def tensor_transforms(v2=False):
    # we convert it to torch.Tensor
    return [A.pytorch.ToTensorV2(True)] if v2 else [A.pytorch.ToTensor()]


def phase_transforms(p=.1):
    def _inv(x, **params):
        return -x
    
    return [A.Lambda(image=_inv, name='inversion', p=p)]


def compose(transforms_list):
    # combine all augmentations into single pipeline
    result = A.Compose([
        item for sublist in transforms_list for item in sublist
    ])
    return result
