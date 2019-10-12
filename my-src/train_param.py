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
from model.gan import Generator, VAEEncoder
from data import ParamDataset

class Trainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader
        self.G = Generator(opt).to(opt.device)
        self.Enc = VAEEncoder(opt).to(opt.device)

        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        self.optimizer_enc = torch.optim.Adam(self.Enc.parameters(), lr=1e-3)
        # self.cirterion = Criterion()
        self.criterion = nn.MSELoss()

    def train(self):
        self.G.train()
        self.Enc.train()

        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (target_im, obj_mask, obj_class, canvas, gt_params) in enumerate(
                self.dataloader
            ):
                target_im = target_im.to(opt.device)
                obj_mask = obj_mask.to(opt.device)
                obj_class = obj_class.to(opt.device)
                canvas = canvas.to(opt.device)
                gt_params = gt_params.to(opt.device)

                z = self.Enc(target_im, obj_mask, obj_class, canvas)
                y = self.G(z, obj_class)  # predicted params
                loss = self.criterion(y, gt_params)

                self.optimizer_g.zero_grad()
                self.optimizer_enc.zero_grad()
                loss.backward()
                self.optimizer_g.step()
                self.optimizer_enc.step()

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
        MyDataset(opt), batch_size=opt.batch_size, shuffle=True
    )

    if opt.train:
        trainer = Trainer(dataloader, opt)
        trainer.train()
