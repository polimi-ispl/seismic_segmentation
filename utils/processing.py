import numpy as np
from scipy.signal import hilbert2
from tensorflow.keras.utils import to_categorical


def chain_functions(x, func_list):
    for f in func_list:
        x = f(x)
    return x


def from_categorical(x):
    return np.argmax(x, axis=-1)


def envelope(in_content: np.ndarray) -> np.ndarray:
    return np.abs(hilbert2(in_content.squeeze()))


def binary_mask(x):
    x[x == 0] = 0
    x[x != 0] = 1
    return x


def shuffle_labels(in_content):
    
    def _shuffle(labels):
        x = np.random.permutation(labels)
        if np.all(x == labels):
            x = _shuffle(labels)
        return x
    
    labels_old = np.unique(in_content)
    labels_new = _shuffle(labels_old)
    
    temp = in_content.copy()
    
    for i, l in enumerate(labels_old):
        in_content[temp == l] = labels_new[i]
    
    return in_content, labels_old, labels_new


def compute_edges(x):
    """BHWC -> BHWC"""
    if x.shape[-1] != 1: # from one-hot to floats
        x = np.argmax(x, axis=-1)
    return np.expand_dims(binary_mask(np.asarray(np.gradient(x, axis=(1, 2))).sum(axis=0)), -1).astype(np.int)


def clip(x, p=95):
    clim = np.percentile(np.abs(x), p)
    return np.clip(x, a_min=-clim, a_max=clim)


def abspower(x, p=1.):
    return np.sign(x) * np.power(np.abs(x), p)


def normalize_01_11(in_content):
    return in_content * 2 - 1


def normalize_only(x, in_min=None, in_max=None, zero_mean=True):
    if in_min is None and in_max is None:
        in_min = np.min(x)
        in_max = np.max(x)
    if np.isclose(in_min, in_max):
        return np.zeros_like(x)
    x = (x - in_min) / (in_max - in_min)
    if zero_mean:
        x = x * 2 - 1
    return x


def normalize_01(in_content):
    in_min, in_max = np.min(in_content), np.max(in_content)
    if np.isclose(in_min, in_max):
        return np.zeros_like(in_content)
    else:
        return (in_content - in_min) / (in_max - in_min)


def png_gamma_correction(in_content, factor=1.):
    in_content = np.power(in_content, factor)
    return np.clip(in_content, 0, 1)


def png_brightness_multiply(in_content, factor=1):
    return np.clip(in_content * factor, 0, 1)


def png_brightness_shift(in_content, factor=0.125):
    return np.clip(in_content + factor, 0, 1)


def remove_mean(in_content):
    return in_content - in_content.mean()


def remove_mean_normalize_std(in_content):
    return (in_content - in_content.mean()) / in_content.std()


def float2png(in_content):
    return np.clip((255 * in_content), 0, 255).astype(np.uint8)


def png2float(in_content):
    return in_content / 255.


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def invert_phase(x, zero_mean=True):
    return -x if zero_mean else (1 - x)


def clip_normalize_power(x, mymin=None, mymax=None, p=1., inv=False):
    """
    Preprocessing function to be applied to migrated images in the C2F scenario

    :param x:       data to be processed
    :param mymin:   min value for clipping
    :param mymax:   max value for clipping
    :param p:       exponent for the power function
    :param inv:     invert the phase
    :return:
    """
    
    if mymin is not None and mymax is not None:
        x = np.clip(x, a_min=mymin, a_max=mymax)
    # normalization can be done if the vector is not constant
    if np.isclose(x.min(), x.max()):
        return np.zeros_like(x)
    else:
        x, _, _ = normalize(x)
    x = np.sign(x) * np.power(np.abs(x), p)
    return x if not inv else -x


def clip_normalize_power_inverse(x, mymin, mymax, p, inv):
    """
    Inverse preprocessing function to be applied to output images in the C2F scenario
    :param inv: invert phase
    :param x: data to be processed
    :param mymin: min value used for clipping
    :param mymax: max value used for clipping
    :param p: exponent for the power function (to be inverted)
    :return:
    """
    if inv:
        x = -x
    x = np.sign(x) * np.power(np.abs(x), 1 / p)
    x = denormalize(x, mymin, mymax)
    return x


def normalize(x, in_min=None, in_max=None, zero_mean=True):
    if in_min is None and in_max is None:
        in_min = np.min(x)
        in_max = np.max(x)
    x = (x - in_min) / (in_max - in_min)
    if zero_mean:
        x = x * 2 - 1
    return x, in_min, in_max


def denormalize(x, in_min, in_max):
    """
    Denormalize data.
    :param x: ndarray, normalized data
    :param in_min: float, the minimum value
    :param in_max: float, the maximum value
    :return: denormalized data in [in_min, in_max]
    """
    if x.min() == 0.:
        return x * (in_max - in_min) + in_min
    else:
        return (x + 1) * (in_max - in_min) / 2 + in_min