import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import numpy as np
from torch.utils.data import DataLoader, dataloader

import config
import dataloaders.dataloaders as dataloaders
import models.models as models
import utils.utils as utils
from dataloaders.MPVDataset import MPVDataset
from dataloaders.VitonDataset import VitonDataset
from dataloaders.VitonHDDataset import VitonHDDataset
from utils.plotter import evaluate, plot_simple_reconstructions

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

def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))
def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader to populate opt ---#
opt.phase = "test"
dataloaders.get_dataloaders(opt)

assert opt.phase in {"val", "test"}

if opt.dataset == "mpv":
    dataset_cl = MPVDataset
elif opt.dataset == "viton":
    dataset_cl = VitonDataset
elif opt.dataset == "vitonHD":
    dataset_cl = VitonHDDataset
else:
    raise NotImplementedError

if (opt.phase == "val" or opt.phase == "test"):
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(opt, model)
    model.eval()
    
    image_indices = [2, 7, 8, 18, 35, 36, 38, 45, 47, 52, 56, 57, 58, 60, 63, 64, 66, 72, 74, 80]

    dataset = dataset_cl(opt, phase=opt.phase)
    evaluate(model, dataset, opt)

if opt.phase == "test":
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(opt, model)
    model.eval()

    dataset = dataset_cl(opt, phase=opt.phase)
    
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    
    os.makedirs(os.path.join("results", opt.name, opt.phase + "_images"), exist_ok=True)
    
    for i, data_i in enumerate(test_dataloader):
        print(i, "/", len(test_dataloader), end="\r")
        image, label, human_parsing = models.preprocess_input(opt, data_i)
        # label["cloth_seg"] = model.module.edit_cloth_seg(image["C_t"], label["body_seg"], label["cloth_seg"])
        agnostic = data_i["agnostic"] if opt.bpgm_id.find("old") >= 0 else None
        
        if opt.no_seg:
            image["I_m"] = image["I"]
        
        pred = model(image, label, "generate", None, agnostic=agnostic).detach().cpu().squeeze().permute(1, 2, 0).numpy()
        # print(pred.shape)
        pred = (pred + 1) / 2
        pred = (pred * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        # pred = cv2.resize(pred, (data_i['original_size'][1], data_i['original_size'][0]), interpolation=cv2.INTER_LINEAR)
        pred = cv2.resize(pred, opt.img_size[::-1], interpolation=cv2.INTER_AREA)
        
        if opt.dataset == "mpv":
            filename = data_i['name'][0].split("/")[-1].replace(".jpg", ".png")
        elif opt.dataset == "viton":
            filename = data_i['name'][0].split("/")[-1]
        elif opt.dataset == "vitonHD":
            filename = data_i['name'][0].split("/")[-1]
        cv2.imwrite(os.path.join("results", opt.name, opt.phase + "_images", filename,"_pred"), pred)

        cv2.imwrite(os.path.join("results", opt.name, opt.phase + "_images", filename, "_origin"), tens_to_im(image["I"][0]))
        cv2.imwrite(os.path.join("results", opt.name, opt.phase + "_images", filename, "_cloth"), tens_to_im(image["C_t"][0]))
        im = tens_to_lab(label["densepose_seg"][0], opt.semantic_nc[2] + 1)
        cv2.imwrite(os.path.join("results", opt.name, opt.phase + "_images", filename, "_densepose"), im)




