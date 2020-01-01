# %%writefile /content/CoordConv/gan-textbox/train_toy_param_comnist_vae.py
import os
from typing import List, Tuple, Union, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from config import get_parameters
from data import ParamDataset
from generator.blocks import CLASSES
from model.gan import VAEEncoder
from model.fc import Regressor, Classifier

"""
前提：detector找出`mask`, `cat`，每個物件拆分為`shape`, `fill`2種類型
目的：
預測`image`中`mask`物件的params，針對給予`shape|fill`來決定是預測shape/fill的params，不是每個物件都需要預測shape params。
照片/Icon的預測要如何做？
    input: (image, mask, cat, shape|fill)
    predict: (params)
"""


class Generator(nn.Module):
    def __init__(self, z_dim, ns_cats: List[int], p_dim, n_bks=len(CLASSES)):
        super().__init__()
        self.n_bks = n_bks
        ns_cats.append(n_bks)  # next_cat as final classifier

        self.embedding = nn.Embedding(n_bks, 32)
        self.mask = nn.Linear(n_bks, p_dim, bias=False)

        x_dim = z_dim + 2 * 32  # x is (z, cat_embed, pre_cat_embed)
        self.rg = Regressor(x_dim, p_dim)
        self.cfs = nn.ModuleList([Classifier(x_dim, n) for n in ns_cats])

    def forward(self, z, cat, pre_cat):
        """
        Args:
            z: (N, n_masks, z_dim), features of image
            cat: (N, 1), class of masked item
            pre_cat: (N, 1)
        Return:
            param: (N, n_params)
            is_cat: [(N, n_classes), ..., (N, n_bks)->next_cat], logits
        """
        x = torch.cat((z, self.embedding(cat), self.embedding(pre_cat)), dim=1)
        mask = self.mask(
            F.one_hot(cat, self.n_bks).float()
        )  # 利用mask class來幫忙regressor推測param
        param = self.rg(x) * mask
        cats = [cf(x) for cf in self.cfs]
        return param, cats


def train(data_loader, opt):
    Enc = VAEEncoder(opt).to(opt.device)
    G = Generator(opt.z_dim, opt.ns_cats, opt.p_dim).to(opt.device)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=1e-3)
    optimizer_enc = torch.optim.Adam(Enc.parameters(), lr=1e-3)
    rg_loss = nn.MSELoss()
    cf_loss = nn.CrossEntropyLoss()

    Enc.train()
    G.train()

    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (im, mask, cat, pre_cat, param, cats) in enumerate(dataloader):
            x = torch.cat((im, mask), dim=1)
            x = x.to(opt.device)
            cat = cat.to(opt.device)
            pre_cat = pre_cat.to(opt.device)
            param = param.to(opt.device)
            cats = (i.to(opt.device) for i in cats)

            z = Enc(x)
            _param, _cats = G(z, cat, pre_cat)  # predicted params

            # print([cf_loss(x, y) for x, y in zip(_cats, cats)])
            _loss = [rg_loss(_param, param)]
            for x, y in zip(_cats, cats):
                _loss += [cf_loss(x, y)]

            loss = sum(_loss)
            _loss = [l.item() for l in _loss]

            optimizer_g.zero_grad()
            optimizer_enc.zero_grad()
            loss.backward()
            optimizer_g.step()
            optimizer_enc.step()

            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [loss: %s]"
                    % (epoch, opt.n_epochs, i, len(dataloader), str(_loss))
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

    ds = ParamDataset("/workspace/CRAFT-pytorch/my-dataset/train.json", cats=CLASSES)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=opt.batch_size, shuffle=True
    )
    opt.ns_cats = ds.meta["ns_cats"]
    opt.p_dim = ds.p_dim

    if opt.train:
        train(dataloader, opt)
