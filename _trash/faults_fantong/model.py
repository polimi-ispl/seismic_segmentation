import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import networks
from util import create_scheduler
import numpy as np

class BaseModel(ABC):
    """This class is an abstract bese class for models.
    To create a subclass, you need to implement the following five fucntions.
        -- <__init__>:  initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>: unpack data form dataset and apply preprocessing.
        -- <forward>:   produce intermdediate results.
        -- <optimize_parameters>:  calculate losses, gradients, and update networks weights.
    """

    def __init__(self, opt):
        """Initialize the BaseModel calss.

        Parameters:
            opt (option class) -- store all the experiment flags.

        When creating your custom claa, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then you need to define your lists:
            -- self.loss_names (str list):  specify the training losses that you want to plot and save.
            -- self.model_names(str list):  define networks used in our training.
            -- self.visual_names (str list): specify the images that you want to display and save.
            -- self.optimizers （optimizer list): define and initialize optimizers.
                You can define one optimizer for each network. If two networks are updated at the
                same time, you can use itertools.chain to group them. See cycle_gan_model.py for example
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.isTrain = opt.isTrain
        self.save_dir = opt.results_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.metric = 0  # only used for learning rate policy 'plateau'
        self.image_paths = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Prameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parametes> and <test>."""
        pass

    @abstractmethod
    def train(self):
        """calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers"""
        if self.isTrain:
            self.schedulers = [create_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            self.load_networks()
        self.print_networks()

    def save_networks(self):
        """Save all the networks to the disk every epoch.
        the name of the checkpoint is latest_net_(model name).pth
        the model is saved in the GPU

        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = 'latest_net%s.pth' % name
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self):
        """
        The function is load all the networks from the disk.
        the name of the checkpoint is latest_net_(model name).pth
        The model is saved in the GPU.

        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = 'latest_net%s.pth' % name
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self):
        """Print the total number of parameters in the network and network architecture to the txt file"""
        with open(os.path.join(self.save_dir, 'net.txt'), "w") as net_txt:
            print('---------- Networks initialized -------------', file=net_txt)
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(self, 'net' + name)
                    num_params = 0
                    for param in net.parameters():
                        num_params += param.numel()
                    print(net, file=net_txt)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6), file=net_txt)
            print('-----------------------------------------------', file=net_txt)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for schedule in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                schedule.step(self.metric)
            else:
                schedule.step()

        self.lr = self.optimizers[0].param_groups[0]['lr']

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def etrain(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don;t save intermediate steps for backprop.
        It also calls <compute_visuals> to produce additional visualization results.
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with vidsom and svae the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return training losses / errors. train.py will print out these errors on console, and save them to a file."""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = getattr(self, 'loss_' + name)
        return errors_ret

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations
        parameters:
            nets (network list)     -- a list of networks
            requires_grad (bool)    -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class BaiSegmentationModel(BaseModel):
    """ The model is for the segmentation of the borehole acoustic image.
    The model training requires '--dataset_mode BaiSegmentation' dataset.
    By default, it uses a '--netG unet128' U-Net generator,
    and Dice losses.

    interpolation paper:
    """
    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['BCE', 'Dice']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['img', 'gen', 'lbl']
        # specify the models you want to save to the disk.
        self.model_names = ['_unet']
        # define networks (both generator and discriminator)
        if opt.model == 'unet':
            self.net_unet = networks.define_unet(opt.input_nc, opt.output_nc, opt.norm, opt.act_fun, opt.need_bias,
                                                 opt.init_type, opt.init_gain, self.gpu_ids)
        elif opt.model == 'multi':
            self.net_unet = networks.define_unet(opt.input_nc, opt.output_nc, 1.67, opt.norm, opt.act_fun, opt.need_bias,
                                                 opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            assert False

        # define loss functions
        self.criterionDice = networks.DiceLoss()
        # self.criterionWCB = networks.weightcrossentropyloss()
        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_unet = torch.optim.Adam(self.net_unet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_unet)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.img = input['img'].float().to(self.device)
        self.lbl = input['lbl'].float().to(self.device)
        self.image_paths = input['img_name']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.gen = self.net_unet(self.img)  # unet(A)
        self.loss_Dice = 1 - self.criterionDice(self.gen, self.lbl)
        self.loss_BCE = self.criterionBCE(self.gen, self.lbl)

    def train(self):
        # self.etrain()
        self.forward()                   # compute fake images: unet(A)
        self.set_requires_grad(self.net_unet, True)  # D requires no gradients when optimizing G
        self.optimizer_unet.zero_grad()        # set G's gradients to zero
        self.loss_BCE.backward()                  # calculate graidents for G
        self.optimizer_unet.step()             # udpate G's weights

    def val(self):
        # self.eval()
        with torch.no_grad():
            self.forward()                   # compute fake images: unet(A)
            # self.loss_Dice = 1 - self.loss_Dice
