import copy
import os

import lpips
import torch
import torch.nn as nn
from bpgm.model.utils import load_checkpoint as bpgm_load
# from seg_edit.utils.utils import load_checkpoint as seg_edit_load
# from seg_edit.model.models import UNet
from torch.nn import functional as F
from torch.nn import init
from torch.cuda.amp import autocast

import models.discriminator as discriminators
import models.generator as generators
import models.losses as losses
from models.sync_batchnorm import DataParallelWithCallback
import numpy as np


class OASIS_model(nn.Module):

    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        # self.netG = generators.OASIS_Generator(opt)
        self.netG = generators.OASIS_Simple(opt)
        if opt.phase in {"train", "train_whole"}:
            if self.opt.add_d_loss:
                self.netD = discriminators.OASIS_Discriminator(opt)
            else:
                self.netD = None
            if self.opt.add_cd_loss:
                self.netCD = discriminators.CDiscriminator(opt)
            if self.opt.add_pd_loss:
                self.netPD = discriminators.PDiscriminator(opt)
            if self.opt.add_hd_loss:
                self.netHD = discriminators.HumanParsingDiscriminator(opt)
        self.print_parameter_count()
        self.init_networks()

        if "cloth" in self.opt.segmentation:
            self.seg_edit = None
            # self.seg_edit = UNet(opt, 3 + (opt.label_nc[0] + 1) + (opt.label_nc[1] - 6 + 1), 6 + 1)
            # self.seg_edit.eval()
            # seg_edit_load(self.seg_edit, "./seg_edit/checkpoints/seg_final_%s.pth" % (opt.seg_edit_id))
        else:
            self.seg_edit = None

        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None

        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        if opt.transform_cloth:
            bpgm_load(self.netG.bpgm, "./bpgm/checkpoints/bpgm_final_%s.pth" % (opt.bpgm_id))

        # --- perceptual loss ---#
        if opt.phase in {"train", "train_whole"}:
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss()
            if opt.add_l1_loss:
                self.L1_loss = nn.L1Loss()
            if opt.add_lpips_loss:
                self.LPIPS_loss = lpips.LPIPS(net="vgg", verbose=False)
            if opt.add_l2_loss:
                self.L2_loss = nn.MSELoss()

            self.entropy_loss = losses.CrossEntropyLoss()

    def forward(self, image, label, mode, losses_computer, label_centroids=None, agnostic=None, human_parsing=None):
        # Branching is applied to be compatible with DataParallel
        with autocast():
            if mode == "losses_G":
                loss_G = 0

                image = generate_swapped_batch(image)

                # cloth_seg = self.edit_cloth_seg(image["C_t_swap"], label["body_seg"], label["cloth_seg"])
                # cloth_seg = self.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])

                # fake now has 19 channels
                fake, C_transform = self.netG(image["I_m"], image["C_t"], image["cloth_mask"], label["body_seg"],
                                              label["cloth_seg"], label["densepose_seg"], agnostic=agnostic,
                                              human_parsing=human_parsing)
                full_fake = fake
                fake = fake[:, 0:3, :, :]
                # from PIL import Image
                # import numpy as np
                fake_target, C_target_transform = self.netG(image["I_m"], image["target_cloth"],
                                                            image["target_cloth_mask"], label["body_seg"],
                                                            label["cloth_seg"], label["densepose_seg"],
                                                            agnostic=agnostic, human_parsing=human_parsing)

                if self.opt.add_d_loss:
                    # fake = self.netG(image["I_m"], image["C_t"], label["body_seg"], label["cloth_seg"], label["densepose_seg"])

                    # DELET AFTER
                    # _fake = ((fake * 0.5 + 0.5).detach()[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    # Image.fromarray(_fake).save(os.path.join("sample", "fake_swap.png"))

                    output_D = self.netD(fake)

                    # DELET AFTER
                    # output_D = F.softmax(output_D, dim=1)
                    # fake_class = (output_D.detach()[0][:1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    # fake_class = np.where(fake_class < 64, 255, 0)
                    # fake_class = np.repeat(fake_class, 3, axis=-1).astype(np.uint8)
                    # Image.fromarray(fake_class).save(os.path.join("sample", "fake_class.png"))
                    # 
                    # fake_label = output_D.detach()[0][1:].permute(1, 2, 0).cpu().numpy()
                    # fake_label = np.argmax(fake_label, axis=-1).astype(np.float32)
                    # fake_label /= fake_label.max()
                    # fake_label = (fake_label * 255).astype(np.uint8)
                    # 
                    # Image.fromarray(fake_label).save(os.path.join("sample", "fake_label.png"))

                    if "body" in self.opt.segmentation:
                        loss_G_adv_D_body = losses_computer.loss(
                            output_D[:, self.opt.offsets[0]:self.opt.offsets[1], :, :], label["body_seg"],
                            for_real=True)
                        loss_G += loss_G_adv_D_body
                    else:
                        loss_G_adv_D_body = None

                    if "cloth" in self.opt.segmentation:
                        loss_G_adv_D_cloth = losses_computer.loss(
                            output_D[:, self.opt.offsets[1]:self.opt.offsets[2], :, :], label["cloth_seg"],
                            for_real=True)
                        loss_G += loss_G_adv_D_cloth
                    else:
                        loss_G_adv_D_cloth = None

                    if "densepose" in self.opt.segmentation:
                        loss_G_adv_D_densepose = losses_computer.loss(
                            output_D[:, self.opt.offsets[2]:self.opt.offsets[3], :, :], label["densepose_seg"],
                            for_real=True)
                        loss_G += loss_G_adv_D_densepose
                    else:
                        loss_G_adv_D_densepose = None
                else:
                    loss_G_adv_D_body, loss_G_adv_D_cloth, loss_G_adv_D_densepose = None, None, None

                if self.opt.add_cd_loss:
                    # output_CD = self.netCD(fake, image["C_t_swap"])
                    output_CD = self.netCD(fake, image["C_t"])
                    loss_G_adv_CD = losses_computer.loss_adv(output_CD, for_real=True)
                    loss_G += loss_G_adv_CD
                else:
                    loss_G_adv_CD = None

                if self.opt.add_hd_loss:
                    output_HD = self.netHD(fake_target[:, 3:, :, :], image["target_cloth_mask"], label["densepose_seg"])
                    loss_G_adv_HD = losses_computer.loss_adv(output_HD, for_real=True)
                    loss_G += loss_G_adv_HD
                else:
                    loss_G_adv_HD = None

                if self.opt.add_pd_loss:
                    fake = generate_patches(self.opt, fake, label_centroids)

                    # for i, fake_sample in enumerate(fake):
                    #     # DELET AFTER
                    #     _fake = ((fake_sample * 0.5 + 0.5).detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    #     Image.fromarray(_fake).save(os.path.join("sample", "patch_%d.png" % i))    

                    output_PD = self.netPD(fake)
                    loss_G_adv_PD = losses_computer.loss_adv(output_PD, for_real=True)
                    loss_G += loss_G_adv_PD
                else:
                    loss_G_adv_PD = None

                image = generate_swapped_batch(image)

                if self.opt.add_vgg_loss or self.opt.add_lpips_loss or self.opt.add_l1_loss:
                    fake, C_transform = self.netG(image["I_m"], image["C_t"], image["cloth_mask"], label["body_seg"],
                                                  label["cloth_seg"], label["densepose_seg"], agnostic=agnostic,
                                                  human_parsing=human_parsing)
                    full_fake = fake
                    fake = fake[:, 0:3, :, :]
                    # DELET AFTER
                    # _fake = ((fake * 0.5 + 0.5).detach()[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    # Image.fromarray(_fake).save(os.path.join("sample", "fake.png"))

                if self.opt.add_vgg_loss:

                    loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image['I'])
                    loss_G += loss_G_vgg
                else:
                    loss_G_vgg = None

                if self.opt.add_l1_loss:
                    loss_G_l1 = self.opt.lambda_l1 * self.L1_loss(fake, image['I'])
                    loss_G_l1_parsing = self.opt.lambda_l1 * self.L1_loss(full_fake[:, 3:, :, :], human_parsing)
                    loss_G += loss_G_l1
                    loss_G += loss_G_l1_parsing
                else:
                    loss_G_l1 = None
                    loss_G_l1_parsing = None

                if self.opt.add_lpips_loss:
                    loss_G_lpips = self.opt.lambda_lpips * self.LPIPS_loss(fake, image['I']).mean()
                    loss_G_lpips_parsing = self.opt.lambda_lpips * self.LPIPS_loss(full_fake[:, 3:, :, :],
                                                                                   human_parsing).mean()
                    loss_G += loss_G_lpips
                    loss_G += loss_G_lpips_parsing
                else:
                    loss_G_lpips = None
                    loss_G_lpips_parsing = None
                if self.opt.add_l2_loss:
                    loss_G_l2 = self.L2_loss(full_fake[:, 3:, :, :], human_parsing)
                    loss_G += loss_G_l2
                else:
                    loss_G_l2 = None

                if self.opt.add_crossEntropy_loss:
                    parsing = torch.argmax(human_parsing, dim=1)
                    fake_parsing = torch.argmax(full_fake[:, 3:, :, :], dim=1)
                    loss_G_parsing = self.entropy_loss(full_fake[:, 3:, :, :], parsing)
                    loss_G += loss_G_parsing
                if self.opt.add_parsing_loss:
                    real_parsing = torch.argmax(human_parsing, dim=1)
                    fake_parsing = torch.argmax(full_fake[:, 3:, :, :], dim=1)

                    losses.CalculateParsingLoss(real_parsing, fake_parsing)
                if self.opt.add_shape_loss:
                    fake_target_parsing = torch.argmax(fake_target[:, 3:, :, :], dim=1)
                    fake_arg_015 = torch.argmax(fake_target[:, [3, 4, 5], :, :], dim=1)
                    fake_target_upper = torch.eq(fake_target_parsing, fake_arg_015).cuda().float()
                    C_target_transform_binary = C_target_transform[:, 0, :, :]
                    mask_target_cloth = torch.zeros(C_target_transform_binary.size(), dtype=torch.uint8).cuda().float()
                    mask_target_cloth[torch.all(C_target_transform_binary > 0)] = 1

                    loss_shape_l2 = self.L2_loss(fake_target_upper, mask_target_cloth)
                    loss_G += loss_shape_l2


                return loss_G, [loss_G_adv_D_body, loss_G_adv_D_cloth, loss_G_adv_D_densepose, loss_G_adv_CD, loss_G_adv_HD,
                                loss_G_adv_PD, loss_G_vgg, loss_G_l1, loss_G_lpips]

            elif mode == "losses_D":
                loss_D = 0

                with autocast(enabled=False):
                    image = generate_swapped_batch(image)

                    # cloth_seg = self.edit_cloth_seg(image["C_t_swap"], label["body_seg"], label["cloth_seg"])
                    # cloth_seg = self.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])

                    with torch.no_grad():
                        # fake = self.netG(image["I_m"], image["C_t_swap"], label["body_seg"], cloth_seg, label["densepose_seg"])
                        fake, C_transform = self.netG(image["I_m"], image["target_cloth"], image["target_cloth_mask"],
                                                      label["body_seg"], label["cloth_seg"], label["densepose_seg"],
                                                      agnostic=agnostic, human_parsing=human_parsing)
                        full_fake = fake
                        fake = fake[:, 0:3, :, :]

                    output_D_fake = self.netD(fake)

                    if "body" in self.opt.segmentation:
                        loss_D_fake_body = losses_computer.loss(
                            output_D_fake[:, self.opt.offsets[0]:self.opt.offsets[1], :, :], label["body_seg"],
                            for_real=False)
                        loss_D += loss_D_fake_body
                    else:
                        loss_D_fake_body = None

                    if "cloth" in self.opt.segmentation:
                        loss_D_fake_cloth = losses_computer.loss(
                            output_D_fake[:, self.opt.offsets[1]:self.opt.offsets[2], :, :], label["cloth_seg"],
                            for_real=False)
                        loss_D += loss_D_fake_cloth
                    else:
                        loss_D_fake_cloth = None

                    if "densepose" in self.opt.segmentation:
                        loss_D_fake_densepose = losses_computer.loss(
                            output_D_fake[:, self.opt.offsets[2]:self.opt.offsets[3], :, :], label["densepose_seg"],
                            for_real=False)
                        loss_D += loss_D_fake_densepose
                    else:
                        loss_D_fake_densepose = None

                    image = generate_swapped_batch(image)

                    output_D_real = self.netD(image['I'])

                    if "body" in self.opt.segmentation:
                        loss_D_real_body = losses_computer.loss(
                            output_D_real[:, self.opt.offsets[0]:self.opt.offsets[1], :, :], label["body_seg"],
                            for_real=True)
                        loss_D += loss_D_real_body
                    else:
                        loss_D_real_body = None

                    if "cloth" in self.opt.segmentation:
                        loss_D_real_cloth = losses_computer.loss(
                            output_D_real[:, self.opt.offsets[1]:self.opt.offsets[2], :, :], label["cloth_seg"],
                            for_real=True)
                        loss_D += loss_D_real_cloth
                    else:
                        loss_D_real_cloth = None

                    if "densepose" in self.opt.segmentation:
                        loss_D_real_densepose = losses_computer.loss(
                            output_D_real[:, self.opt.offsets[2]:self.opt.offsets[3], :, :], label["densepose_seg"],
                            for_real=True)
                        loss_D += loss_D_real_densepose
                    else:
                        loss_D_real_densepose = None

                    if not self.opt.no_labelmix:
                        mixed_inp, mask = generate_labelmix(label, fake, image['I'])

                        output_D_mixed = self.netD(mixed_inp)
                        loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                             output_D_fake,
                                                                                             output_D_real)
                        loss_D += loss_D_lm
                    else:
                        loss_D_lm = None

                return loss_D, [loss_D_fake_body, loss_D_fake_cloth, loss_D_fake_densepose, loss_D_real_body,
                                loss_D_real_cloth, loss_D_real_densepose, loss_D_lm]

            elif mode == "losses_CD":
                loss_CD = 0

                image = generate_swapped_batch(image)

                # cloth_seg = self.edit_cloth_seg(image["C_t_swap"], label["body_seg"], label["cloth_seg"])
                cloth_seg = self.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])

                with torch.no_grad():
                    # fake = self.netG(image["I_m"], image["C_t_swap"], label["body_seg"], cloth_seg, label["densepose_seg"])
                    fake, C_transform = self.netG(image["I_m"], image["target_cloth"], image["target_cloth_mask"],
                                                  label["body_seg"], cloth_seg, label["densepose_seg"],
                                                  agnostic=agnostic)
                    full_fake = fake
                    fake = fake[:, 0:3, :, :]

                # output_CD_fake = self.netCD(fake, image["C_t_swap"])
                output_CD_fake = self.netCD(fake, image["target_cloth"])
                loss_CD_fake = losses_computer.loss_adv(output_CD_fake, for_real=False)
                loss_CD += loss_CD_fake

                image = generate_swapped_batch(image)

                output_CD_real = self.netCD(image['I'], image["C_t"])
                loss_CD_real = losses_computer.loss_adv(output_CD_real, for_real=True)
                loss_CD += loss_CD_real

                return loss_CD, [loss_CD_fake, loss_CD_real]

            elif mode == "losses_PD":
                loss_PD = 0

                image = generate_swapped_batch(image)

                # cloth_seg = self.edit_cloth_seg(image["C_t_swap"], label["body_seg"], label["cloth_seg"])
                cloth_seg = self.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])

                with torch.no_grad():
                    # fake = self.netG(image["I_m"], image["C_t_swap"], label["body_seg"], cloth_seg, label["densepose_seg"])
                    fake, C_transform = self.netG(image["I_m"], image["target_cloth"], image["target_cloth_mask"],
                                                  label["body_seg"], cloth_seg, label["densepose_seg"],
                                                  agnostic=agnostic)
                    full_fake = fake
                    fake = fake[:, 0:3, :, :]

                fake = generate_patches(self.opt, fake, label_centroids)
                output_PD_fake = self.netPD(fake)
                loss_PD_fake = losses_computer.loss_adv(output_PD_fake, for_real=False)
                loss_PD += loss_PD_fake

                image = generate_swapped_batch(image)

                image_patches = generate_patches(self.opt, image["I"], label_centroids)
                output_PD_real = self.netPD(image_patches)
                loss_PD_real = losses_computer.loss_adv(output_PD_real, for_real=True)
                loss_PD += loss_PD_real

                return loss_PD, [loss_PD_fake, loss_PD_real]
            elif mode == "losses_HD":
                loss_HD = 0
                image = generate_swapped_batch(image)
                cloth_seg = self.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])
                with torch.no_grad():
                    fake, C_transform = self.netG(image["I_m"], image["target_cloth"], image["target_cloth_mask"],
                                                  label["body_seg"], cloth_seg, label["densepose_seg"],
                                                  agnostic=agnostic)
                    fake_parsing = fake[:, 3:, :, :]
                output_HD_fake = self.netHD(fake_parsing, image["target_cloth_mask"], label["densepose_seg"])#
                loss_HD_fake = losses_computer.loss_adv(output_HD_fake, for_real=False)
                loss_HD += loss_HD_fake
                output_HD_real = self.netHD(label["target_parsing"], image["target_cloth_mask"], label['densepose_seg_target'])
                loss_HD_real = losses_computer.loss_adv(output_HD_real, for_real=True)
                loss_HD += loss_HD_real
                return loss_HD, [loss_HD_fake, loss_HD_real]
            elif mode == "generate":
                with torch.no_grad():
                    if self.opt.no_EMA:
                        fake, C_transform = self.netG(image["I_m"], image["C_t"], image["cloth_mask"],
                                                      label["body_seg"], label["cloth_seg"], label["densepose_seg"],
                                                      agnostic=agnostic, human_parsing=human_parsing)
                        full_fake = fake
                        fake = fake[:, 0:3, :, :]
                    else:
                        fake, C_transform = self.netEMA(image["I_m"], image["C_t"], image["cloth_mask"],
                                                        label["body_seg"], label["cloth_seg"], label["densepose_seg"],
                                                        agnostic=agnostic, human_parsing=human_parsing)
                        full_fake = fake
                        fake = fake[:, 0:3, :, :]
                return fake

            else:
                raise NotImplementedError

    def load_checkpoints(self):
        if self.opt.phase == "test" or self.opt.phase == "val":
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(self.opt.which_iter) + "_")
            print(f"checkpoint path {path}")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"), strict=False)
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"), strict=False)
        elif self.opt.continue_train:
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(self.opt.which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))

            if self.opt.add_d_loss:
                self.netD.load_state_dict(torch.load(path + "D.pth"))

            if self.opt.add_cd_loss:
                self.netCD.load_state_dict(torch.load(path + "CD.pth"))

            if self.opt.add_pd_loss:
                self.netPD.load_state_dict(torch.load(path + "PD.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def edit_cloth_seg(self, C_t, body_seg, cloth_seg):
        if self.seg_edit is not None:
            with torch.no_grad():
                cloth_seg = torch.clone(cloth_seg)
                if self.seg_edit.resolution != self.opt.img_size:
                    _C_t = F.interpolate(C_t, size=self.seg_edit.resolution, mode="bilinear", align_corners=False)
                    _body_seg = F.interpolate(body_seg, size=self.seg_edit.resolution, mode="nearest")
                    _cloth_seg = F.interpolate(cloth_seg, size=self.seg_edit.resolution, mode="nearest")

                    x = torch.cat((_C_t, _body_seg, _cloth_seg[:, 6:, :, :]), dim=1)

                    # convert to one-hot
                    upper_cloth_seg = torch.argmax(self.seg_edit(x)[:, :6, :, :], dim=1, keepdim=True)
                    upper_cloth_seg_one_hot = torch.zeros(
                        (upper_cloth_seg.shape[0], 6, *self.seg_edit.resolution)).cuda()
                    upper_cloth_seg_one_hot = upper_cloth_seg_one_hot.scatter(1, upper_cloth_seg, 1.0)

                    cloth_seg[:, :6, :, :] = F.interpolate(upper_cloth_seg_one_hot, size=self.opt.img_size,
                                                           mode="nearest")
                else:
                    x = torch.cat((C_t, body_seg, cloth_seg[:, 6:, :, :]), dim=1)

                    # convert to one-hot
                    upper_cloth_seg = torch.argmax(self.seg_edit(x)[:, :6, :, :], dim=1, keepdim=True)
                    upper_cloth_seg_one_hot = torch.zeros(
                        (upper_cloth_seg.shape[0], 6, *self.seg_edit.resolution)).cuda()
                    upper_cloth_seg_one_hot = upper_cloth_seg_one_hot.scatter(1, upper_cloth_seg, 1.0)

                    cloth_seg[:, :6, :, :] = self.seg_edit(x)[:, :6, :, :]

            return cloth_seg
        else:
            return torch.clone(cloth_seg)

    def print_parameter_count(self):
        if self.opt.phase in {"train", "train_whole"} and self.opt.add_d_loss:
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase in {"train", "train_whole"} and self.opt.add_d_loss:
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(opt, model):
    model = DataParallelWithCallback(model, device_ids=opt.gpu_ids).cuda()

    # assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['cloth_label'] = data['cloth_label'].long()
    data['body_label'] = data['body_label'].long()
    data['densepose_label'] = data['densepose_label'].long()
    data['densepose_label_target'] = data['densepose_label_target'].long()
    data['human_parsing'] = data['human_parsing'].long()
    data['human_parsing_target'] = data['human_parsing_target'].long()

    data['cloth_label'] = data['cloth_label'].cuda()
    data['body_label'] = data['body_label'].cuda()
    data['densepose_label'] = data['densepose_label'].cuda()
    data['densepose_label_target'] = data['densepose_label_target'].cuda()
    data['human_parsing'] = data['human_parsing'].cuda()
    data['human_parsing_target'] = data['human_parsing_target'].cuda()

    for key in data['image'].keys():
        data['image'][key] = data['image'][key].cuda()

    label_body_map = data['body_label']
    bs, _, h, w = label_body_map.size()
    nc = opt.semantic_nc[0]  # 16
    input_body_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_body_semantics = input_body_label.scatter_(1, label_body_map, 1.0)

    label_cloth_map = data['cloth_label']
    bs, _, h, w = label_cloth_map.size()
    nc = opt.semantic_nc[1]  # 16
    input_cloth_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_cloth_semantics = input_cloth_label.scatter_(1, label_cloth_map, 1.0)

    label_densepose_map = data['densepose_label']
    bs, _, h, w = label_densepose_map.size()
    nc = opt.semantic_nc[2]  # 26
    input_densepose_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_densepose_semantics = input_densepose_label.scatter_(1, label_densepose_map, 1.0)

    label_densepose_map_target = data['densepose_label_target']
    bs, _, h, w = label_densepose_map_target.size()
    nc = opt.semantic_nc[2]  # 26
    input_densepose_label_target = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_densepose_semantics_target = input_densepose_label_target.scatter_(1, label_densepose_map_target, 1.0)

    human_parsing_map = data["human_parsing"]
    bs, _, h, w = human_parsing_map.size()
    nc = 16
    input_human_parsing_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_human_parsing_semantics = input_human_parsing_label.scatter_(1, human_parsing_map, 1.0)

    human_parsing_target_map = data["human_parsing_target"]
    bs, _, h, w = human_parsing_target_map.size()
    nc = 16
    input_human_parsing_target_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_human_parsing_target_semantics = input_human_parsing_target_label.scatter_(1, human_parsing_target_map, 1.0)

    return data['image'], {"body_seg": input_body_semantics, "cloth_seg": input_cloth_semantics,
                           "densepose_seg": input_densepose_semantics,"densepose_seg_target": input_densepose_semantics_target, "target_parsing":input_human_parsing_target_semantics}, input_human_parsing_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to(label.device)
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map


def generate_patches(opt, images, label_centroids):
    patches = []
    for centroid in label_centroids:
        x = centroid[0]
        y = centroid[1]

        for _x, _y, im in zip(x, y, images):
            if _x != -1 and _y != -1:
                x_min = max(_x - opt.patch_size // 2, 0)
                x_max = x_min + opt.patch_size
                if x_max > opt.img_size[0]:
                    x_max = opt.img_size[0]
                    x_min = x_max - opt.patch_size

                y_min = max(_y - opt.patch_size // 2, 0)
                y_max = y_min + opt.patch_size
                if y_max > opt.img_size[1]:
                    y_max = opt.img_size[1]
                    y_min = y_max - opt.patch_size

                patch = im[:, x_min:x_max, y_min:y_max]
                patches.append(patch.unsqueeze(0))

    patches = torch.cat(patches, dim=0)
    return patches


def generate_swapped_batch(image):
    # image["C_t"] = image["C_t"].flip(0)
    return image


def Parsing2rgb(parsing):
    cmap = np.array([  # 15
        [254, 85, 0],  # top
        [0, 0, 85],  # one piece
        [0, 85, 85],  # pants
        [0, 128, 0],  # skirt
        [0, 119, 220],  # jacket
        [254, 169, 0],  # left foot
        [254, 254, 0],  # right foot
        [0, 0, 0],  # background
        [254, 0, 0],  # hair
        [0, 0, 254],  # face
        [0, 254, 254],  # right arm
        [51, 169, 220],  # left arm
        [85, 51, 0],  # torso
        [169, 254, 85],  # right leg
        [85, 254, 169],  # left leg
    ], dtype=np.uint8)
    cmap = torch.from_numpy(cmap[:17])
    size = parsing.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    parsing = torch.argmax(parsing, dim=0, keepdim=True)
    for label in range(0, len(cmap)):
        mask = (label == parsing[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image
