import argparse
import os
import torch
import util
import time


class Options():
    """This class define options used during both training and test time.

    It also implements several helper function such as parsing, printing, and saving the options.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # dataset parameters
        parser.add_argument('--dataroot', required=True,
                            help='path to images (should have subfolders train and test. the data should be alaigned)')
        parser.add_argument('--dataset_name', type=str, default='Seismicnp',
                            help='the class name of the dataset, the first letter should be capitalized')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes image in order to make batches, otherwise tkes them randomly')
        parser.add_argument('--num_threads', type=int,
                            default=0, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int,
                            default=1, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'),
                            help='Maximum number of samples allowed per dataset, if the dataset contains more data,'
                            'then a subset will be extracted')
        parser.add_argument('--use_val', action='store_false',
                            help='use the validation mode in the training')
        parser.add_argument('--val_split', type=float, default=0.25,
                            help='the validation percent in the trainging data. used only if use_val')
        parser.add_argument('--num_trval', type=float, default=0.8,
                            help='the percentation of the images used for train and val')

        # model parameters
        parser.add_argument('--model', type=str, default='Unet',
                            help='the name of the network')
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--norm', type=str, default='batch',
                            help='normalization, [instance | batch | none]')
        parser.add_argument('--act_fun', type=str, default='ReLU',
                            help='activate function, [LeakyReLU | ELU | ReLU | Tanh]')
        parser.add_argument('--init_type', default='normal',
                            help='network initialization [normal| xavier| kaiming| orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--need_bias', action='store_true',
                            help='no dropout for the generator')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU')

        # training parameters
        parser.add_argument('--description', type=str, default='the first for interpolation',
                            help='The brief description of the experiment')
        parser.add_argument('--phase', type=str, default='train',
                            help='the mode of the code, [ train| test |val, etc,.]')
        parser.add_argument('--niter', type=int, default=40, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=60, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=20,
                            help='multiply by a gamma every lr_decay_iters iterations use for step')

        # test parameters
        parser.add_argument('--ntest', type=int,
                            default=float('inf'), help='# of test examples.')
        # Dropout and Batchnorm has different behavior during training and test.
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float('inf'),
                            help='how many test images to run')

        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=30,
                            help='numer of iters of showing training results on screen')
        parser.add_argument('--display_id', type=int, default=1,
                            help='window id of the web display')
        parser.add_argument('--display_server', type=str, default='http://localhost',
                            help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display enviroment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8098,
                            help='visdom port of the web display')
        parser.add_argument('--display_winsize', type=int, default=180,
                            help='display window size for both visdom and HTML')

        # saving folders
        parser.add_argument('--results_dir', type=str, default='./results/',
                            help='save model, image, loss of train, val, test here. \
                                a date folder will be made in this results.')
        parser.add_argument('--date', type=str, default='',
                            help='the date foler of the results dir. if not define, will make by the now time.')
        # network save and load
        parser.add_argument('--save_epoch_freq', type=int, default=5,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by '
                            '<epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--print_freq', type=int, default=30,
                            help='number of iter of showing training result on console.')
        # load trained model parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cahed model')
        parser.add_argument('--verbose', action='store_false',
                            help='if specified, print more debugging information')

        # patch parameters
        parser.add_argument('--patch_stride', type=tuple,
                            default=(128, 128), help='the stride of regular patch')
        parser.add_argument('--patch_shape', type=tuple,
                            default=(128, 128), help='the shape of regular patch')
        parser.add_argument('--patch_number', type=int,
                            default=64, help='the number of the random extracted patchs')
        parser.add_argument('--patch_random', action='store_true',
                            help='the number of the random extracted patchs')
        parser.add_argument('--gain', type=float, default=2000,
                            help='the number multiple to the patchs')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once)."""
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        phase = opt.phase
        if phase == 'train':
            self.isTrain = True
        else:
            self.isTrain = False

        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """print and save options

        It will print both current options and default values(if different).
        It will save options to a text file/ [restults_dir]/opt_'phase'.txt
        """
        message = ''
        message += '---------------------- Options ------------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk

        file_name = os.path.join(opt.results_dir, 'opt_' + opt.phase + '.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, unit_test=False):
        """Parse our options, create result directory, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # define the results dir.

        if opt.date == '':
            opt.results_dir = os.path.join(opt.results_dir, time.strftime("%Y-%m-%d-%H", time.localtime()))
            util.mkdirs(opt.results_dir)
        else:
            opt.results_dir = os.path.join(opt.results_dir, opt.date)
            if not os.path.exists(opt.results_dir):
                raise Exception('The results folder is not exist.')

        if not unit_test:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            torch.backends.cudnn.benchmark = False

        self.opt = opt
        self.opt.hypara = sorted(vars(opt).items())
        return self.opt
