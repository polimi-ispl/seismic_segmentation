import os
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from utils.processing import from_categorical, normalize


def plot_label(x, fig, ax, num_classes=2, labels=None, cmap='tab10', **imshow_args):
    color_map = cm.get_cmap(cmap, num_classes)
    
    clim = (0, num_classes - 1)
    
    mask = ax.imshow(x, cmap=color_map, clim=clim, **imshow_args)
    
    if labels is not None:
        assert len(labels) == num_classes
        cbar = fig.colorbar(mask, ax=ax)
        cbar_ticks = np.linspace(clim[0] + .5, clim[-1] - .5, num_classes)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.set_yticklabels(labels)


def plot_triplets(inputs, targets, outputs, title, outpath, num_classes, labels, is_categorical=True):
    
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(16, 8))
    
    axs[0].imshow(inputs.squeeze())
    axs[0].set_title('inputs')
    axs[0].axis('tight')
    
    if not is_categorical:
        axs[1].imshow(targets.squeeze(), vmin=0, vmax=1)
    else:
        plot_label(from_categorical(targets.squeeze()), fig, axs[1], num_classes,
                   cmap='tab10', **dict(alpha=1.))
    axs[1].set_title('targets')
    axs[1].axis('tight')
    
    if not is_categorical:
        axs[2].imshow(outputs.squeeze(), vmin=0, vmax=1)
    else:
        plot_label(from_categorical(outputs.squeeze()), fig, axs[2], num_classes,
                   labels=labels, cmap='tab10', **dict(alpha=1.))
    axs[2].set_title('outputs')
    axs[2].axis('tight')
    
    fig.suptitle(title)
    plt.savefig(os.path.join(outpath, title + '.png'))
    # plt.show()
    

def plot_training(history, outpath):
    plt.figure(figsize=(12, 4))
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch'), plt.ylabel('Loss')
    plt.xlim(0), plt.ylim(0)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(outpath, 'curves.png'))


def clim(in_content, ratio=95):
    c = np.percentile(np.absolute(in_content), ratio)
    return -c, c


def image_mask_superposed(name: str, image: np.ndarray, mask: np.ndarray, figsize=(8, 8), p=95, alpha=.1, colorbar_image=True):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray', clim=clim(image, p))
    if colorbar_image:
        plt.colorbar(label='amplitude', aspect=50)
    plt.imshow(mask, cmap='jet', alpha=alpha)
    if not colorbar_image:
        plt.colorbar(label='classes', aspect=50)
    plt.title(name)
    plt.tight_layout(pad=.5)
    plt.show()


def save_image(in_content, filename, clim=(None, None), folder='./'):
    """
    Save a gray-scale PNG image of the 2D content

    :param in_content:  2D np.ndarray
    :param filename:    name of the output file (without extension)
    :param clim:        tuple for color clipping (as done in matplotlib.pyplot.imshow)
    :param folder:      output directory
    :return:
    """
    if clim[0] and clim[1] is not None:
        in_content = np.clip(in_content, clim[0], clim[1])
        in_content = normalize(in_content, in_min=clim[0], in_max=clim[1])[0]
    else:
        in_content = normalize(in_content)[0]
    out = Image.fromarray(((in_content + 1) / 2 * 255).astype(np.uint8))
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    out.save(os.path.join(folder, filename + '.png'))


def plot2pgf(temp, filename, folder='./'):
    """
    :param temp:        list of equally-long data
    :param filename:    filename without extension nor path
    :param folder:      folder where to save
    """
    if len(temp) == 1:  # if used as plt.plot(y) without the abscissa axis
        temp = [list(range(len(temp[0]))), temp[0]]

    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T,
               fmt="%f", encoding='ascii')