# %%writefile /content/CoordConv/gan-textbox/train_toy_param_comnist_vae.py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from config import get_parameters
from model.gan import VAEEncoder
from data import ParamDataset
from generator.blocks import CLASSES


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # self.opt = opt
        self.imsize = opt.imsize
        self.device = opt.device
        self.class_dim = len(CLASSES)  # mask class
        regressor_dim = 4

        # self.model = self._classifier()
        # regressor mask, 利用mask class來幫忙regressor推測param
        self.rg_mask = nn.Linear(self.class_dim, regressor_dim, bias=False)
        self.rg = nn.Sequential(
            nn.Linear(opt.z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, regressor_dim),
        )

    def _classifier(self, z_dim, n_classes):
        return nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, z, cat):
        """
        Args:
            z: (N, n_masks, z_dim), features of image
            cat: (N, n_labels), mask class, one hot
        Return: (N, )
        """
        cat = F.one_hot(cat, self.class_dim).float()
        mask = self.rg_mask(cat)
        return self.rg(z) * mask


class Trainer(object):
    def __init__(self, data_loader, opt):
        self.opt = opt
        self.dataloader = dataloader
        self.Enc = VAEEncoder(opt).to(opt.device)
        self.G = Generator(opt).to(opt.device)

        self.optimizer_g = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        self.optimizer_enc = torch.optim.Adam(self.Enc.parameters(), lr=1e-3)
        # self.cirterion = Criterion()
        self.criterion = nn.MSELoss()

    def train(self):
        self.Enc.train()
        self.G.train()

        batches_done = 0
        for epoch in range(opt.n_epochs):
            for i, (im, mask, cat, gt_param) in enumerate(self.dataloader):
                # print(gt_param)
                # print(im, mask, cat, gt_param)
                # im = im.to(opt.device)
                # mask = mask.to(opt.device)
                x = torch.cat((im, mask), dim=1)
                x = x.to(opt.device)
                cat = cat.to(opt.device)
                gt_param = gt_param.to(opt.device)

                z = self.Enc(x)
                y = self.G(z, cat)  # predicted params
                loss = self.criterion(y, gt_param)

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
                    # save_image(
                    #     # denormalize(fake_img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).data[
                    #     #     :25
                    #     # ],
                    #     y.data[:9],
                    #     os.path.join(
                    #         self.opt.sample_path, "{:06d}_fake.png".format(batches_done)
                    #     ),
                    #     nrow=5,
                    #     # normalize=True,
                    # )
                batches_done += 1


if __name__ == "__main__":
    opt = get_parameters()
    opt.cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.img_shape = (opt.im_channels, opt.imsize, opt.imsize)

    os.makedirs(opt.model_save_path, exist_ok=True)
    os.makedirs(opt.sample_path, exist_ok=True)
    os.makedirs(opt.log_path, exist_ok=True)

    if opt.cuda:
        torch.backends.cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(
        ParamDataset(opt), batch_size=opt.batch_size, shuffle=True
    )

    if opt.train:
        trainer = Trainer(dataloader, opt)
        trainer.train()
