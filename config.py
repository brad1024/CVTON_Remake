import argparse
import pickle
import os
import utils.utils as utils
import math


def read_arguments(train=True, whole=False):
    parser = argparse.ArgumentParser()
    parser = add_all_arguments(parser, train)
    parser.add_argument('--phase', type=str, default='train')
    
    opt = parser.parse_args()
    set_dataset_default_lm(opt, parser)
    if train:
        if opt.continue_train:
            update_options_from_file(opt, parser)
    
    opt = parser.parse_args()
    opt.phase = 'train' if train else 'test'
    opt.phase += "_whole" if whole else ""
    
    assert all([kword in {"body", "cloth", "densepose"} for kword in opt.segmentation])
    opt.segmentation = sorted(opt.segmentation)
    
    if train:
        opt.loaded_latest_iter = 0 if not opt.continue_train else load_iter(opt)
    
    utils.fix_seed(opt.seed)
    
    print_options(opt, parser)
    
    if train:
        save_options(opt, parser)
    
    return opt


def add_all_arguments(parser, train):
    #--- general options ---
    parser.add_argument('--name', "-n", type=str, help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--no_spectral_norm', action='store_true', help='this option deactivates spectral norm in all layers')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--dataset', type=str, default="mpv", help="Dataset to use.")
    parser.add_argument('--dataroot', type=str, default='./data/mpv/', help='path to dataset root')
    parser.add_argument('--img_size', type=int, default=256, help='image size.')
    parser.add_argument('--which_iter', type=str, default='latest', help='which epoch to load when continue_train')
    parser.add_argument('--gpu_ids', nargs='+', default=[0], type=int, help="GPUs to use for training / inference")

    # for generator
    parser.add_argument('--num_res_blocks', type=int, default=6, help='number of residual blocks in G and D')
    parser.add_argument('--channels_G', type=int, default=32, help='# of gen filters in first conv layer in generator')
    parser.add_argument('--param_free_norm', type=str, default='batch', help='which norm to use in generator before SPADE')
    parser.add_argument('--spade_ks', type=int, default=3, help='kernel size of convs inside SPADE')
    parser.add_argument('--no_EMA', action='store_true', help='if specified, do *not* compute exponential moving averages')
    parser.add_argument('--EMA_decay', type=float, default=0.9999, help='decay in exponential moving averages')
    parser.add_argument('--z_dim', type=int, default=12, help="dimension of the latent z vector")
    
    parser.add_argument('--val_size', type=float, default=0.05, help="Validation set size (fraction - not int).")
    parser.add_argument('--train_size', type=float, default=0.95, help="Train set size (fraction - not int).")
    parser.add_argument('--transform_cloth', action='store_true', help="Whether to feed a transformed cloth to the OASIS architecture.")
    parser.add_argument('--bpgm_id', type=str, default="256_3_5", help="BPGM identification for pretrained weights loading.")
    parser.add_argument('--seg_edit_id', type=str, default="256", help="SEG_EDIT model identification for pretrained weights loading.")
    parser.add_argument('--segmentation', nargs='+', default=["body"], help="Which segmentations to use for conditioning. {body, cloth, densepose}")

    parser.add_argument('--no_seg', action='store_true', default=False, help='whether to train the model without masking clothing')
    parser.add_argument("--no_bg", action='store_true', default=False, help="whether to remove the background in I_m")

    if train:
        parser.add_argument('--freq_print', type=int, default=100, help='frequency of showing training results')
        parser.add_argument('--freq_save_ckpt', type=int, default=2000, help='frequency of saving the checkpoints')
        parser.add_argument('--freq_save_latest', type=int, default=1000, help='frequency of saving the latest model')
        parser.add_argument('--freq_smooth_loss', type=int, default=250, help='smoothing window for loss visualization')
        parser.add_argument('--freq_save_loss', type=int, default=2000, help='frequency of loss plot updates')
        parser.add_argument('--freq_fid', type=int, default=5000, help='frequency of saving the fid score (in training iterations)')
        parser.add_argument('--continue_train', action='store_true', help='resume previously interrupted training')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr_g', type=float, default=0.0001, help='G learning rate, default=0.0001')
        parser.add_argument('--lr_d', type=float, default=0.0004, help='D learning rate, default=0.0004')

        parser.add_argument('--channels_D', type=int, default=64, help='# of discrim filters in first conv layer in discriminator')
        parser.add_argument('--add_vgg_loss', action='store_true', help='if specified, add VGG feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=1.0, help='weight for VGG loss')
        parser.add_argument('--add_lpips_loss', action='store_true', help='if specified, add LPIPS feature matching loss')
        parser.add_argument('--lambda_lpips', type=float, default=1.0, help='weight for LPIPS loss')
        parser.add_argument('--add_l1_loss', action='store_true', help='if specified, add L1 loss')
        parser.add_argument('--add_l2_loss', action='store_true', help='if specified, add L2 loss')
        parser.add_argument('--add_crossEntropy_loss', action='store_true', help='if specified, add cross entropy loss')
        parser.add_argument('--add_parsing_loss', action='store_true', help='if specified, add parsing loss')
        parser.add_argument('--add_shape_loss', action='store_true', help='if specified, add shape loss')
        parser.add_argument('--lambda_l1', type=float, default=1.0, help='weight for L1 loss')
        parser.add_argument('--add_d_loss', action="store_true", help="if specified, add segmentation discriminator loss")
        parser.add_argument('--add_cd_loss', action="store_true", help="if specified, add conditional discriminator loss")
        parser.add_argument('--add_pd_loss', action="store_true", help="if specified, add patch discriminator loss")
        parser.add_argument('--add_hd_loss', action="store_true", help="if specified, add human parsing discriminator loss")
        parser.add_argument('--patch_size', type=int, default=0, help="patch size for patch discriminator")
        parser.add_argument('--no_balancing_inloss', action='store_true', default=False, help='if specified, do *not* use class balancing in the loss function')
        parser.add_argument('--no_labelmix', action='store_true', default=False, help='if specified, do *not* use LabelMix')
        parser.add_argument('--lambda_labelmix', type=float, default=10.0, help='weight for LabelMix regularization')
        
    else:
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves testing results here.')
        
    return parser


def set_dataset_default_lm(opt, parser):
    if opt.dataset == "mpv":
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(num_res_blocks=int(math.log(opt.img_size, 2)) - 2)
        parser.set_defaults(dataroot="./data/mpv")
        parser.set_defaults(patch_size=opt.img_size // 4)
    elif opt.dataset == "viton":
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(num_res_blocks=int(math.log(opt.img_size, 2)) - 2)
        parser.set_defaults(dataroot="./data/viton")
        parser.set_defaults(patch_size=opt.img_size // 4)
    elif opt.dataset == "vitonHD":
        parser.set_defaults(num_epochs=100)
        parser.set_defaults(num_res_blocks=int(math.log(opt.img_size, 2)) - 2)
        parser.set_defaults(dataroot="./data/vitonHD")
        parser.set_defaults(patch_size=opt.img_size // 4)
    else:
        raise NotImplementedError


def save_options(opt, parser):
    path_name = os.path.join(opt.checkpoints_dir,opt.name)
    os.makedirs(path_name, exist_ok=True)
    with open(path_name + '/opt.txt', 'wt') as opt_file:
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

    with open(path_name + '/opt.pkl', 'wb') as opt_file:
        pickle.dump(opt, opt_file)


def update_options_from_file(opt, parser):
    new_opt = load_options(opt)
    for k, v in sorted(vars(opt).items()):
        if hasattr(new_opt, k) and v != getattr(new_opt, k):
            new_val = getattr(new_opt, k)
            parser.set_defaults(**{k: new_val})
    return parser


def load_options(opt):
    file_name = os.path.join(opt.checkpoints_dir, opt.name, "opt.pkl")
    new_opt = pickle.load(open(file_name, 'rb'))
    return new_opt


def load_iter(opt):
    if opt.which_iter == "latest":
        with open(os.path.join(opt.checkpoints_dir, opt.name, "latest_iter.txt"), "r") as f:
            res = int(f.read())
            return res
    else:
        return int(opt.which_iter)


def print_options(opt, parser):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
