import torch
import numpy as np
import random
import time
import os
import models.models as models
import matplotlib.pyplot as plt
from PIL import Image


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter = (start_iter + 1) % dataset_size
    return start_epoch, start_iter


class results_saver():
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.which_iter)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt):
        self.name_list = ["Generator_D_body", "Generator_D_cloth", "Generator_D_densepose", "Generator_CD", "Generator_HD",
                          "Generator_PD", "Vgg", "L1", "LPIPS", "D_fake_body", "D_fake_cloth", "D_fake_densepose",
                          "D_real_body", "D_real_cloth", "D_real_densepose", "LabelMix", "CD_fake", "CD_real",
                          "PD_fake", "PD_real"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path + "/losses.npy", allow_pickle=True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()

        if epoch % self.freq_smooth_loss == 0:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i] / self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0

        if epoch % self.freq_save_loss == 0:
            self.plot_losses()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig, ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve]))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (curve)), dpi=600)
            plt.close(fig)

        fig, ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    # update weights based on new generator weights
    with torch.no_grad():
        for key in model.module.netEMA.state_dict():
            model.module.netEMA.state_dict()[key].data.copy_(
                model.module.netEMA.state_dict()[key].data * opt.EMA_decay +
                model.module.netG.state_dict()[key].data * (1 - opt.EMA_decay)
            )
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                image, label, human_parsing = models.preprocess_input(opt, data_i)
                agnostic = data_i["agnostic"].cuda() if opt.bpgm_id.find("old") >= 0 else None
                fake, C_transform = model.module.netEMA(image["I_m"], image["C_t"], image["cloth_mask"], label["body_seg"], label["cloth_seg"], label["densepose_seg"], agnostic=agnostic)
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        torch.save(model.module.netG.state_dict(), path + '/%s_G.pth' % ("latest"))
        if opt.add_d_loss:
            torch.save(model.module.netD.state_dict(), path + '/%s_D.pth' % ("latest"))

        if opt.add_cd_loss:
            torch.save(model.module.netCD.state_dict(), path + '/%s_CD.pth' % ("latest"))

        if opt.add_pd_loss:
            torch.save(model.module.netPD.state_dict(), path + '/%s_PD.pth' % ("latest"))

        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%s_EMA.pth' % ("latest"))

        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        torch.save(model.module.netG.state_dict(), path + '/%s_G.pth' % ("best"))
        if opt.add_d_loss:
            torch.save(model.module.netD.state_dict(), path + '/%s_D.pth' % ("best"))

        if opt.add_cd_loss:
            torch.save(model.module.netCD.state_dict(), path + '/%s_CD.pth' % ("best"))

        if opt.add_pd_loss:
            torch.save(model.module.netPD.state_dict(), path + '/%s_PD.pth' % ("best"))

        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%s_EMA.pth' % ("best"))

        with open(os.path.join(opt.checkpoints_dir, opt.name) + "/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        torch.save(model.module.netG.state_dict(), path + '/%d_G.pth' % (cur_iter))
        if opt.add_d_loss:
            torch.save(model.module.netD.state_dict(), path + '/%d_D.pth' % (cur_iter))

        if opt.add_cd_loss:
            torch.save(model.module.netCD.state_dict(), path + '/%d_CD.pth' % (cur_iter))

        if opt.add_pd_loss:
            torch.save(model.module.netPD.state_dict(), path + '/%d_PD.pth' % (cur_iter))

        if not opt.no_EMA:
            torch.save(model.module.netEMA.state_dict(), path + '/%d_EMA.pth' % (cur_iter))


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
        self.opt = opt
        self.num_cl = {"body_seg": opt.semantic_nc[0] + 1, "cloth_seg": opt.semantic_nc[1] + 1,
                       "densepose_seg": opt.semantic_nc[2] + 1, "fake_original_parsing": 17, "real_parsing": 17, "fake_target_parsing" : 17}

        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter, agnostic=None, human_parsing=None):
        if "body" in self.opt.segmentation:
            self.save_images(label["body_seg"], "body_seg", cur_iter, is_label=True)
        if "cloth" in self.opt.segmentation:
            self.save_images(label["cloth_seg"], "cloth_seg", cur_iter, is_label=True)
        if "densepose" in self.opt.segmentation:
            self.save_images(label["densepose_seg"], "densepose_seg", cur_iter, is_label=True)
        # print('-------------------------------------')
        # print(label["densepose_seg"])
        # print('--dense---------------00000----------')
        # time.sleep(30)
        self.save_images(image['I'], "real", cur_iter)
        self.save_images(human_parsing, "real_parsing", cur_iter, is_label=True)
        self.save_images(image['target_cloth'], "fake_target_cloth", cur_iter)
        with torch.no_grad():
            model.eval()
            fake, C_transform = model.module.netG(image["I_m"], image["C_t"], image["cloth_mask"], label["body_seg"], label["cloth_seg"],
                                     label["densepose_seg"], agnostic=agnostic)
            fake_parsing = fake[:, 3:, :, :]
            fake = fake[:, 0:3, :, :]
            fake_target, C_target_transform = model.module.netG(image["I_m"], image["target_cloth"], image["target_cloth_mask"], label["body_seg"], label["cloth_seg"],
                                            label["densepose_seg"], agnostic=agnostic)
            fake_target_parsing = fake_target[:, 3:, :, :]
            fake_target = fake_target[:, 0:3, :, :]
            self.save_images(fake, "fake_original", cur_iter)
            self.save_images(fake_parsing, "fake_original_parsing", cur_iter, is_label=True)
            self.save_images(C_transform, "fake_original_cloth_transform", cur_iter)
            self.save_images(fake_target, "fake_target", cur_iter)
            self.save_images(fake_target_parsing, "fake_target_parsing", cur_iter, is_label=True)
            self.save_images(C_target_transform, "fake_target_cloth_transform", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                fake, C_transform = model.module.netEMA(image["I_m"], image["C_t"], image["cloth_mask"], label["body_seg"], label["cloth_seg"],
                                           label["densepose_seg"], agnostic=agnostic)
                fake_parsing = fake[:, 3:, :, :]
                fake = fake[:, 0:3, :, :]
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl[name])
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path + str(cur_iter) + "_" + name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    # print('==========N===========')
    # print(N)
    # time.sleep(30)
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    elif N == 27:
        # print('-----N-----27-----')
        # time.sleep(30)
        cmap = np.array([  # 25+1
            [0, 0, 0],
            [20, 80, 194],
            [20, 80, 194],
            [9, 109, 221],
            [4, 98, 224],
            [12, 123, 215],
            [20, 133, 213],
            [3, 167, 195],
            [26, 174, 188],
            [6, 166, 198],
            [22, 174, 184],
            [120, 189, 135],
            [86, 187, 143],
            [115, 189, 128],
            [88, 186, 145],
            [145, 191, 116],
            [170, 190, 103],
            [191, 188, 111],
            [216, 189, 86],
            [252, 207, 46],
            [250, 220, 34],
            [254, 206, 46],
            [240, 191, 52],
            [251, 235, 25],
            [247, 252, 12],
            [200, 100, 200],
            [100, 200, 200]
        ], dtype=np.uint8)

    elif N == 17:
        cmap = np.array([  # 15
            [254, 85, 0],  # top
            [0, 0, 85],  # one piece
            [85, 51, 0],  # torso
            [0, 254, 254],  # right arm
            [51, 169, 220],  # left arm
            [0, 119, 220],  # jacket
            [0, 0, 0],  # background
            [0, 85, 85],  # pants
            [254, 0, 0],  # hair
            [0, 128, 0],  # skirt
            [254, 169, 0],  # left foot
            [254, 254, 0],  # right foot
            [0, 0, 254],  # face
            [169, 254, 85],  # right leg
            [85, 254, 169],  # left leg
        ], dtype=np.uint8)

    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
