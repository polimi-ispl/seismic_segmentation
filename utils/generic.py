"""
@Author: Francesco Picetti - francesco.picetti@polimi.it
"""
import numpy as np
import os
from socket import gethostname
import json
from pathlib import Path
import random
import string
from GPUtil import getFirstAvailable, getGPUs


# Metrics

def mse(target, output):
    return np.mean((target - output) ** 2)


def snr(target, output):
    """
    Compute SNR between the target and the reconstructed images

    :param target:  numpy array of reference
    :param output:  numpy array we have produced
    :return: SNR in dB
    """
    if target.shape != output.shape:
        raise ValueError('There is something wrong with the dimensions!')
    return 20 * np.log10(np.linalg.norm(target) / np.linalg.norm(target - output))


# Utilities

def flip_coin():
    return bool(random.getrandbits(1))


def set_data_path(host='polimi'):
    if host == 'polimi':
        data_path = '/nas/home/fpicetti/geophysics/datasets'
    elif host == 'cineca':
        data_path = '/gpfs/scratch/usera06ptm/a06ptm04/fpicetti/datasets'
    else:
        raise ValueError('host name not recognized!')
    return data_path


def log10plot(in_content):
    return np.log10(np.asarray(in_content) / in_content[0])


def ten_digit(number):
    return int(np.floor(np.log10(number)) + 1)


def int2str(in_content, digit_number):
    in_content = int(in_content)
    return str(in_content).zfill(ten_digit(digit_number))


def random_code(n=6):
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(int(n))])


def machine_name():
    return gethostname()


def idle_cpu_count(mincpu=1):
    # the load is computed over the last 1 minute
    idle = int(os.cpu_count() - np.floor(os.getloadavg()[0]))
    return max(mincpu, idle)


def save_args_to_file(outpath, args):
    os.makedirs(outpath, exist_ok=True)
    with open(outpath / 'args.txt', 'w') as fp:
        a = args.__dict__.copy()
        for k in a.keys():
            if isinstance(a[k], Path):
                a[k] = str(a[k])
        json.dump(a, fp, indent=2)


def sec2time(seconds):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def nextpow2(x):
    return int(2 ** np.ceil(np.log2(x)))


def prevpow2(x):
    return int(2**np.floor(np.log2(x)))


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)
    """
    if id is None:
        print('GPU not selected')
    else:
        try:
            device = id if id != -1 else getFirstAvailable(order='memory')[0]  # -1 for automatic choice
        except RuntimeError:
            print('WARNING! No GPU available, switching to CPU')
            return
        try:
            name = getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most '
                  'available one.')
            device = getFirstAvailable(order='memory')[0]
            name = getGPUs()[device].name
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        print('GPU selected: %d - %s' % (device, name))
