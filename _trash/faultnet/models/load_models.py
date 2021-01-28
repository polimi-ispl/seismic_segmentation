import os,sys
import torch

from .vnet import VNet
from .vnet_modified import VNet_modified
from .unet_3D_Res1 import unet_3D_Res1
from .unet_3D_DS import unet_3D_DS

def getModel(model_arch, output_channels=2, parallel_flag=False, gpu_flag=True,base_filter=32):
   
    print(" model arch specified by user is ",model_arch)

    if model_arch.lower() == 'vnet':
        print("loading VNet, output will have %d channels"%(output_channels))
        model = VNet()

    elif model_arch.lower() == 'vnet_modified':
        print("loading VNet Modified, output will have %d channels"%(output_channels))
        model = VNet_modified()

    elif model_arch.lower() == 'unet_3d_res1':
        print("loading UNet3d with 1 Residual block, output will have %d channels"%(output_channels))
        model = unet_3D_Res1()

    elif model_arch.lower() == 'unet_3d_ds':
        print("loading UNet3d with Deep supervision  output will have base-filter size %d"%(base_filter))
        model = unet_3D_DS(n_classes=output_channels, base_n_filter=base_filter)

    else:
        
        error_mess = "Model arch not implemented !!"
        raise ValueError(error_mess)

    if parallel_flag==True:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if gpu_flag==True:
        model.cuda()

    return model

        
