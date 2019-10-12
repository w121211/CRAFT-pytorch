# %%writefile /content/CoordConv/gan-textbox/train_toy_param_comnist_vae.py
import os
import glob
import random
import time
import datetime
from collections import OrderedDict

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageColor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# from models.craft import CRAFTGenerator
# from models.wgan import Generator, Discriminator, compute_gradient_penalty
from config import get_parameters

# from utils import tensor2var, denorm
from model.craft import CRAFT

# from loss import Criterion
from data import MaskDataset


def denormalize(x, mean, std):
    dtype = x.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=x.device)
    std = torch.as_tensor(std, dtype=dtype, device=x.device)
    std_inv = 1 / (std + 1e-7)
    mean_inv = -mean * std_inv
    x.sub_(mean_inv[None, :, None, None]).div_(std_inv[None, :, None, None])
    return x


class Trainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader
        self.model = CRAFT(opt).to(opt.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        # self.cirterion = Criterion()

    def train(self):
        self.model.train()

        # iterator = tqdm(dataloader)
        # def change_lr(no_i):
        #     for i in config.lr:
        #         if i == no_i:
        #             print("Learning Rate Changed to ", config.lr[i])
        #             for param_group in optimizer.param_groups:
        #                 param_group["lr"] = config.lr[i]

        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (image, cur_masks, target_mask) in enumerate(self.dataloader):
                # change_lr(no)
                x = torch.cat([image, cur_masks], dim=1)
                x = x.to(opt.device)
                target_mask = target_mask.to(opt.device)

                y = self.model(x)
                loss = F.binary_cross_entropy(torch.sigmoid(y), target_mask)
                # loss = self.cirterion(y, target_mask)

                # loss = (
                #     loss_criterian(output, weight, weight_affinity).mean()
                #     / config.optimizer_iteration
                # )
                # all_loss.append(loss.item() * config.optimizer_iteration)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batches_done % opt.sample_interval == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                        % (epoch, opt.n_epochs, i, len(dataloader), loss.item())
                    )
                    save_image(
                        # denormalize(fake_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).data[
                        #     :25
                        # ],
                        y.data[:9],
                        os.path.join(
                            self.opt.sample_path, "{:06d}_fake.png".format(batches_done)
                        ),
                        nrow=5,
                        # normalize=True,
                    )
                batches_done += 1


if __name__ == "__main__":
    opt = get_parameters()
    opt.cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.img_shape = (opt.im_channels, opt.imsize, opt.imsize)

    os.makedirs(opt.model_save_path, exist_ok=True)
    os.makedirs(opt.sample_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.attn_path, exist_ok=True)

    if opt.cuda:
        torch.backends.cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(
        MaskDataset(opt), batch_size=opt.batch_size, shuffle=True
    )

    if opt.train:
        trainer = Trainer(dataloader, opt)
        trainer.train()
