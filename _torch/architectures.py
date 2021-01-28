import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, functional as F
from collections import OrderedDict


def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


def init_weights(network, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        network (network) -- network to be initialized
        init_type (str)   -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float) -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    
    def init_func(m):  # define a initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 10.0, init_gain * 10)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    network.apply(init_func)  # apply the initialization function <init_func>


def init_net(network, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights

    Parameters:
        network (network)  -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        network.to(gpu_ids[0])
        network = torch.nn.DataParallel(network, gpu_ids)
    init_weights(network, init_type, init_gain=init_gain)
    return network


# DoubleConv Class to perform two layer Convolution
class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, nf=64, maxpool=2):
        super(UNet, self).__init__()
        self.conv1 = _DoubleConv(in_ch, nf)
        self.pool1 = nn.MaxPool2d(maxpool)
        self.conv2 = _DoubleConv(nf, nf*(2**1))
        self.pool2 = nn.MaxPool2d(maxpool)
        self.conv3 = _DoubleConv(nf*(2**1), nf*(2**2))
        self.pool3 = nn.MaxPool2d(maxpool)
        self.conv4 = _DoubleConv(nf*(2**2), nf*(2**3))
        self.pool4 = nn.MaxPool2d(maxpool)
        self.conv5 = _DoubleConv(nf*(2**3), nf*(2**4))
        self.up6 = nn.ConvTranspose2d(nf*(2**4), nf*(2**3), 2, stride=2)
        self.conv6 = _DoubleConv(nf*(2**4), nf*(2**3))
        self.up7 = nn.ConvTranspose2d(nf*(2**3), nf*(2**2), 2, stride=2)
        self.conv7 = _DoubleConv(nf*(2**3), nf*(2**2))
        self.up8 = nn.ConvTranspose2d(nf*(2**2), nf*(2**1), 2, stride=2)
        self.conv8 = _DoubleConv(nf*(2**2), nf*(2**1))
        self.up9 = nn.ConvTranspose2d(nf*(2**1), nf, 2, stride=2)
        self.conv9 = _DoubleConv(nf*(2**1), nf*(2**1))
        self.conv9 = _DoubleConv(nf*(2**1), nf)
        self.conv10 = nn.Conv2d(nf, out_ch, 1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Softmax()(c10)
        return out


class _Concat(nn.Module):
    def __init__(self, dim, *args):
        super(_Concat, self).__init__()
        self.dim = dim
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, x):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(x))
        
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])
        
        return torch.cat(inputs_, dim=self.dim)
    
    def __len__(self):
        return len(self._modules)


def _act(act_fun='LeakyReLU'):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        else:
            assert False
    else:
        return act_fun()


def _bn(num_features):
    return nn.BatchNorm2d(num_features)


def _conv(in_f, out_f, kernel_size, stride=1, bias=True):
    """The convolutional filter with same zero pad."""
    to_pad = int((kernel_size - 1) / 2)
    
    convolver = nn.Conv2d(in_f, out_f, kernel_size,
                          stride, padding=to_pad, bias=bias)
    
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)


def _conv2dbn(in_f, out_f, kernel_size, stride=1, bias=True, act_fun='LeakyReLU'):
    block = _conv(in_f, out_f, kernel_size, stride=stride, bias=bias)
    block.add(_bn(out_f))
    block.add(_act(act_fun))
    return block


class _MultiResBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        super(_MultiResBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = _conv2dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1, bias=bias,
                                  act_fun=act_fun)
        self.conv3x3 = _conv2dbn(f_in, int(W * 0.167), 3, 1, bias=bias, act_fun=act_fun)
        self.conv5x5 = _conv2dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias, act_fun=act_fun)
        self.conv7x7 = _conv2dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias, act_fun=act_fun)
        
        self.bn1 = _bn(self.out_dim)
        self.bn2 = _bn(self.out_dim)
        
        self.accfun = _act(act_fun)
    
    def forward(self, x):
        out1 = self.conv3x3(x)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], axis=1))
        out = torch.add(self.shortcut(x), out)
        out = self.bn2(self.accfun(out))
        return out


class _PathRes(nn.Module):
    def __init__(self, f_in, f_out, length, act_fun='LeakyReLU', bias=True):
        super(_PathRes, self).__init__()
        self.network = []
        self.network.append(_conv2dbn(f_in, f_out, 3, 1, bias=bias, act_fun=act_fun))
        self.network.append(_conv2dbn(f_in, f_out, 1, 1, bias=bias, act_fun=act_fun))
        self.network.append(_bn(f_out))
        for i in range(length - 1):
            self.network.append(_conv2dbn(f_out, f_out, 3, 1, bias=bias, act_fun=act_fun))
            self.network.append(_conv2dbn(f_out, f_out, 1, 1, bias=bias, act_fun=act_fun))
            self.network.append(_bn(f_out))
        self.accfun = _act(act_fun)
        self.length = length
        self.network = nn.Sequential(*self.network)
    
    def forward(self, x):
        out = self.network[2](self.accfun(torch.add(self.network[0](x),
                                                    self.network[1](x))))
        for i in range(1, self.length):
            out = self.network[i * 3 + 2](self.accfun(torch.add(self.network[i * 3](out),
                                                                self.network[i * 3 + 1](out))))
        
        return out


def MultiResUnet(nc_in=1, nc_out=1, nc_down=[16, 32, 64, 128, 256], nc_up=[16, 32, 64, 128, 256],
                 nc_skip=[16, 32, 64, 128],
                 alpha=1.67, need_sigmoid=True, need_bias=True, upsample_mode='nearest', act_fun='LeakyReLU'):
    """ The 2D multi-resolution Unet

    Arguments:
        nc_in (int) -- The channels of the input data.
        nc_out (int) -- The channels of the output data.
        nc_down (list) -- The channels of differnt layer in the encoder of networks.
        nc_up (list) -- The channels of differnt layer in the decoder of networks.
        nc_skip (list) -- The channels of path residual block corresponding to different layer.
        alpha (float) -- the value multiplying to the number of filters.
        need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
        need_bias (Bool) -- If add the bias in every convolutional filters.
        upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
        act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
    """
    assert len(nc_down) == len(
        nc_up) == (len(nc_skip) + 1)
    
    n_scales = len(nc_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    multires = _MultiResBlock(nc_down[0], nc_in,
                              alpha=alpha, act_fun=act_fun, bias=need_bias)
    
    model_tmp.add(multires)
    input_depth = multires.out_dim
    
    for i in range(1, len(nc_down)):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        # multi-res Block in the encoders
        multires = _MultiResBlock(nc_down[i], input_depth,
                                  alpha=alpha, act_fun=act_fun, bias=need_bias)
        # stride downsampling.
        deeper.add(_conv(input_depth, input_depth, 3, stride=2, bias=need_bias))
        deeper.add(_bn(input_depth))
        deeper.add(_act(act_fun))
        deeper.add(multires)
        
        if nc_skip[i - 1] != 0:
            # add the path residual block, note that the number of filters is set to 1.
            skip.add(_PathRes(input_depth, nc_skip[i - 1], 1, act_fun=act_fun, bias=need_bias))
            model_tmp.add(_Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        deeper_main = nn.Sequential()
        
        if i != len(nc_down) - 1:
            # not the deepest
            deeper.add(deeper_main)
        # add upsampling to the decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add multi-res block to the decoder
        model_tmp.add(_MultiResBlock(nc_up[i - 1], multires.out_dim + nc_skip[i - 1],
                                     alpha=alpha, act_fun=act_fun, bias=need_bias))
        
        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = nc_up[0] * alpha
    last_kernal = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    # add the convolutional filter for output.
    model.add(
        _conv(last_kernal, nc_out, 1, bias=need_bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model


class ChannelGata(nn.Module):
    """
        The channel block. process the feature extracted by encoder and decoder.
        (Convolutional block attention module)
        referenc (squeeze-and-excitation network)
    """
    
    def __init__(self, f_x, reduction_ratio=4):
        super(ChannelGata, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.psi = nn.Sequential(
            nn.Conv2d(f_x, f_x // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_x // reduction_ratio, f_x, kernel_size=1, stride=1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_max = self.psi(self.maxpool(x))
        x_avg = self.psi(self.avgpool(x))
        return x * self.sigmoid(x_max + x_avg)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    """
        The channel block. process the feature extracted by encoder and decoder.
        (Convolutional block attention module)
        referenc (squeeze-and-excitation network)
    """
    
    def __init__(self, f_x, kernel_size=7):
        super(SpatialGate, self).__init__()
        kernel_size = kernel_size
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        return x * x_out


class CBAM(nn.Module):
    """
        The convolutional block attention module
    """
    
    def __init__(self, f_x, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGata(f_x, reduction_ratio)
        self.SpatialGate = SpatialGate(f_x, kernel_size=kernel_size)
    
    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class Identidy(nn.Module):
    def __init__(self):
        super(Identidy, self).__init__()
    
    def forward(self, x):
        return x


def attention(f_x, kind='unet', reduce_ratio=8, kernel_size=7):
    if kind == 'cbam':
        return CBAM(f_x, reduction_ratio=reduce_ratio, kernel_size=kernel_size)
    else:
        return Identidy()


class AttentionUnet(nn.Module):
    def __init__(self, fin=3, fout=1, act_fun='LeakyReLU', need_bias=True, att='cbam', reduce_ratio=4):
        super(AttentionUnet, self).__init__()
        self.downblock1 = nn.Sequential(OrderedDict({
            'conv1': _conv2dbn(fin, 16, 3, 1, need_bias, act_fun),
            'conv2': _conv2dbn(16, 16, 3, 1, need_bias, act_fun),
            'att1' : attention(16, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 16 X H X W
        self.downblock2 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : _conv2dbn(16, 32, 3, 1, need_bias, act_fun),
            'conv2'   : _conv2dbn(32, 32, 3, 1, need_bias, act_fun),
            'att2'    : attention(32, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 32 X H/2 X W/2
        self.downblock3 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : _conv2dbn(32, 64, 3, 1, need_bias, act_fun),
            'conv2'   : _conv2dbn(64, 64, 3, 1, need_bias, act_fun),
            'att3'    : attention(64, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 64 X H/4 X W/4
        self.downblock4 = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : _conv2dbn(64, 128, 3, 1, need_bias, act_fun),
            'conv2'   : _conv2dbn(128, 128, 3, 1, need_bias, act_fun),
            'att4'    : attention(128, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))  # B X 128 X H/8 X W/8
        self.bottleneck = nn.Sequential(OrderedDict({
            'downconv': nn.MaxPool2d(2, 2),
            'conv1'   : _conv2dbn(128, 256, 3, 1, need_bias, act_fun),
            'conv2'   : _conv2dbn(256, 256, 3, 1, need_bias, act_fun),
            'upconv'  : nn.Upsample(scale_factor=2, mode='bilinear')
        }))  # B X 128 X H/8 X W/8
        
        self.upblock4 = nn.Sequential(OrderedDict({
            'conv1' : _conv2dbn(256 + 128, 128, 3, 1, need_bias, act_fun),
            'conv2' : _conv2dbn(128, 128, 3, 1, need_bias, act_fun),
            'att5'  : attention(128, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear')
        }))
        
        self.upblock3 = nn.Sequential(OrderedDict({
            'conv1' : _conv2dbn(128 + 64, 64, 3, 1, need_bias, act_fun),
            'conv2' : _conv2dbn(64, 64, 3, 1, need_bias, act_fun),
            'att6'  : attention(64, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear')
        }))
        
        self.upblock2 = nn.Sequential(OrderedDict({
            'conv1' : _conv2dbn(64 + 32, 32, 3, 1, need_bias, act_fun),
            'conv2' : _conv2dbn(32, 32, 3, 1, need_bias, act_fun),
            'att7'  : attention(32, kind=att, reduce_ratio=reduce_ratio, kernel_size=7),
            'upconv': nn.Upsample(scale_factor=2, mode='bilinear'),
        }))
        
        self.upblock1 = nn.Sequential(OrderedDict({
            'conv1': _conv2dbn(32 + 16, 16, 3, 1, need_bias, act_fun),
            'conv2': _conv2dbn(16, 16, 3, 1, need_bias, act_fun),
            'att8' : attention(16, kind=att, reduce_ratio=reduce_ratio, kernel_size=7)
        }))
        
        self.outblock = nn.Conv2d(16, fout, 3, 1, 1)
    
    def forward(self, x):
        down1 = self.downblock1(x)
        down2 = self.downblock2(down1)
        down3 = self.downblock3(down2)
        down4 = self.downblock4(down3)
        up4 = self.bottleneck(down4)
        up3 = self.upblock4(torch.cat((down4, up4), dim=1))
        up2 = self.upblock3(torch.cat((down3, up3), dim=1))
        up1 = self.upblock2(torch.cat((down2, up2), dim=1))
        out = self.outblock(self.upblock1(torch.cat((down1, up1), dim=1)))
        
        return out


# For the 3D network
class _Concat3D(nn.Module):
    def __init__(self, dim, *args):
        super(_Concat3D, self).__init__()
        self.dim = dim
        
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)
    
    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))
        
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        inputs_shapes4 = [x.shape[4] for x in inputs]
        
        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)) and np.all(
            np.array(inputs_shapes4) == min(inputs_shapes4)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)
            target_shape4 = min(inputs_shapes4)
            
            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                diff4 = (inp.size(4) - target_shape4) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3, diff4:diff4 + target_shape4])
        
        return torch.cat(inputs_, dim=self.dim)
    
    def __len__(self):
        return len(self._modules)


def _conv3d(in_f, out_f, kernel_size, stride=1, bias=True):
    """
        The 3D convolutional filters with kind of stride, avg pooling, max pooling.
        Note that the padding is zero padding.
    """
    to_pad = int((kernel_size - 1) / 2)
    
    convolver = nn.Conv3d(in_f, out_f, kernel_size,
                          stride, padding=to_pad, bias=bias)
    
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)


def _conv3dbn(in_f, out_f, kernel_size=3, stride=1, bias=True, act_fun='LeakyReLU'):
    block = []
    block.append(_conv3d(in_f, out_f, kernel_size, stride=stride, bias=bias))
    block.append(nn.InstanceNorm3d(out_f))
    block.append(_act(act_fun))
    return nn.Sequential(*block)


class _MultiRes3dBlock(nn.Module):
    def __init__(self, U, f_in, alpha=1.67, act_fun='LeakyReLU', bias=True):
        super(_MultiRes3dBlock, self).__init__()
        W = alpha * U
        self.out_dim = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.shortcut = _conv3dbn(f_in, int(W * 0.167) + int(W * 0.333) + int(W * 0.5), 1, 1,
                                  bias=bias, act_fun=act_fun)
        self.conv3x3 = _conv3dbn(f_in, int(W * 0.167), 3, 1, bias=bias,
                                 act_fun=act_fun)
        self.conv5x5 = _conv3dbn(int(W * 0.167), int(W * 0.333), 3, 1, bias=bias,
                                 act_fun=act_fun)
        self.conv7x7 = _conv3dbn(int(W * 0.333), int(W * 0.5), 3, 1, bias=bias,
                                 act_fun=act_fun)
        self.bn1 = nn.InstanceNorm3d(self.out_dim)
        self.bn2 = nn.InstanceNorm3d(self.out_dim)
        self.accfun = _act(act_fun)
    
    def forward(self, input):
        out1 = self.conv3x3(input)
        out2 = self.conv5x5(out1)
        out3 = self.conv7x7(out2)
        out = self.bn1(torch.cat([out1, out2, out3], axis=1))
        out = torch.add(self.shortcut(input), out)
        out = self.bn2(self.accfun(out))
        return out


class _PathRes3d(nn.Module):
    def __init__(self, f_in, f_out, act_fun='LeakyReLU', bias=True):
        super(_PathRes3d, self).__init__()
        self.conv3x3 = _conv3dbn(f_in, f_out, 3, 1, bias=bias, act_fun=act_fun)
        self.conv1x1 = _conv3dbn(f_in, f_out, 1, 1, bias=bias, act_fun=act_fun)
        self.bn = nn.InstanceNorm3d(f_out)
        self.accfun = _act(act_fun)
    
    def forward(self, input):
        out = self.bn(self.accfun(torch.add(self.conv1x1(input),
                                            self.conv3x3(input))))
        return out


class GridAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(GridAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            _conv3d(F_g, F_int, 1, 1),
            nn.InstanceNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            _conv3d(F_l, F_int, 3, 2),
            nn.InstanceNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            _conv3d(F_int, 1, 1, 1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='trilinear')
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


def MultiResUnet3D(nc_in=1, nc_out=3, nc_down=[16, 32, 64, 128, 256], nc_up=[16, 32, 64, 128, 256],
                   nc_skip=[16, 32, 64, 128],
                   alpha=1.67, need_sigmoid=False, need_bias=True, upsample_mode='nearest', act_fun='LeakyReLU'):
    """
        The 3D multi-resolution Unet
    Arguments:
        nc_in (int) -- The channels of the input data.
        nc_out (int) -- The channels of the output data.
        nc_down (list) -- The channels of differnt layer in the encoder of networks.
        nc_up (list) -- The channels of differnt layer in the decoder of networks.
        nc_skip (list) -- The channels of path residual block corresponding to different layer.
        alpha (float) -- the value multiplying to the number of filters.
        need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
        need_bias (Bool) -- If add the bias in every convolutional filters.
        upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
        act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
    """
    assert len(nc_down) == len(
        nc_up) == (len(nc_skip) + 1)
    
    n_scales = len(nc_down)
    
    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    
    last_scale = n_scales - 1
    
    model = nn.Sequential()
    model_tmp = model
    multires = _MultiRes3dBlock(nc_down[0], nc_in,
                                alpha=alpha, act_fun=act_fun, bias=need_bias)
    
    model_tmp.add(multires)
    input_depth = multires.out_dim
    
    for i in range(1, len(nc_down)):
        
        deeper = nn.Sequential()
        skip = nn.Sequential()
        # add the multi-res block for encoder
        multires = _MultiRes3dBlock(nc_down[i], input_depth,
                                    alpha=alpha, act_fun=act_fun, bias=need_bias)
        # add the stride downsampling for encoder
        deeper.add(_conv3d(input_depth, input_depth, 3, stride=2, bias=need_bias))
        deeper.add(nn.InstanceNorm3d(input_depth))
        deeper.add(_act(act_fun))
        deeper.add(multires)
        
        if nc_skip[i - 1] != 0:
            # add the Path residual block with skip-connection
            skip.add(_PathRes3d(input_depth, nc_skip[i - 1], act_fun=act_fun, bias=need_bias))
            model_tmp.add(_Concat3D(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        deeper_main = nn.Sequential()
        
        if i != len(nc_down) - 1:
            deeper.add(deeper_main)
        # add the upsampling to decoder
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        # add the multi-res block to decoder
        model_tmp.add(_MultiRes3dBlock(nc_up[i - 1], multires.out_dim + nc_skip[i - 1],
                                       alpha=alpha, act_fun=act_fun, bias=need_bias))
        
        input_depth = multires.out_dim
        model_tmp = deeper_main
    W = nc_up[0] * alpha
    last_kernal = int(W * 0.167) + int(W * 0.333) + int(W * 0.5)
    # convolutional filter for output
    model.add(
        _conv3d(last_kernal, nc_out, 1, bias=need_bias))
    if need_sigmoid:
        model.add(nn.Sigmoid())
    
    return model


class AttMulResUnet3D(nn.Module):
    """
        The attention multi-resolution network
    """
    
    def __init__(self, num_input_channels=1, num_output_channels=3,
                 num_channels_down=[16, 32, 64, 128, 256],
                 alpha=1.67, need_sigmoid=False, need_bias=True,
                 upsample_mode='nearest', act_fun='LeakyReLU') -> None:
        """
            The 3D multi-resolution Unet
        Arguments:
            num_input_channels (int) -- The channels of the input data.
            num_output_channels (int) -- The channels of the output data.
            num_channels_down (list) -- The channels of differnt layer in the encoder of networks.
            num_channels_up (list) -- The channels of differnt layer in the decoder of networks.
            num_channels_skip (list) -- The channels of path residual block corresponding to different layer.
            alpha (float) -- the value multiplying to the number of filters.
            need_sigmoid (Bool) -- if add the sigmoid layer in the last of decoder.
            need_bias (Bool) -- If add the bias in every convolutional filters.
            upsample_mode (str) -- The type of upsampling in the decoder, including 'bilinear' and 'nearest'.
            act_fun (str) -- The activate function, including LeakyReLU, ReLU, Tanh, ELU.
        """
        super(AttMulResUnet3D, self).__init__()
        n_scales = len(num_channels_down)
        
        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode = [upsample_mode] * n_scales
        
        input_depths = [num_input_channels]
        
        for i in range(n_scales):
            mrb = _MultiRes3dBlock(num_channels_down[i], input_depths[-1])
            input_depths.append(mrb.out_dim)
            setattr(self, 'down_mb%d' % (i + 1), mrb)
        
        for i in range(1, n_scales):
            setattr(self, 'down%d' % i, nn.Sequential(*[
                _conv3d(input_depths[i], input_depths[i], 3, stride=2, bias=need_bias),
                nn.InstanceNorm3d(input_depths[i]),
                _act(act_fun)
            ]))
            mrb = _MultiRes3dBlock(num_channels_down[-(i + 1)], input_depths[-i] + input_depths[-(i + 1)])
            setattr(self, 'up_mb%d' % i, mrb)
            setattr(self, 'att%d' % i,
                    GridAttentionBlock(input_depths[-i],
                                       input_depths[-(i + 1)], num_channels_down[-i]))
            setattr(self, 'up%d' % i, nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
        if need_sigmoid:
            self.outconv = nn.Sequential(*[
                _conv3d(input_depths[1], num_output_channels, 1, 1, bias=need_bias),
                nn.Sigmoid()])
        else:
            self.outconv = _conv3d(input_depths[1], num_output_channels, 1, 1, bias=need_bias)
    
    def forward(self, inp):
        x1 = self.down_mb1(inp)
        x2 = self.down_mb2(self.down1(x1))
        x3 = self.down_mb3(self.down2(x2))
        x4 = self.down_mb4(self.down3(x3))
        x5 = self.down_mb5(self.down4(x4))
        
        x4 = self.up_mb1(torch.cat([self.att1(x5, x4), self.up1(x5)], dim=1))
        x3 = self.up_mb2(torch.cat([self.att2(x4, x3), self.up2(x4)], dim=1))
        x2 = self.up_mb3(torch.cat([self.att3(x3, x2), self.up3(x3)], dim=1))
        x1 = self.up_mb4(torch.cat([self.att4(x2, x1), self.up4(x2)], dim=1))
        
        return self.outconv(x1)


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=nf,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=nf,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=nf,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs


if __name__ == '__main__':
    inp = torch.rand(1, 1, 64, 64, 64)
    net = AttMulResUnet3D(1, 1)
    print(net)
    out = net(inp)
