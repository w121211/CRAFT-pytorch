import glob
import json

import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image

import torch
import torchvision.transforms as transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.imsize = opt.imsize
        self.n_layers = opt.n_layers
        self.trans_im = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
            ]
        )
        self.trans_mask = transforms.Compose([transforms.ToTensor()])

        with open("/workspace/CRAFT-pytorch/my-dataset/train.json") as f:
            data = json.load(f)
        self.data = data
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplementedError()


class MaskDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.trans_target_mask = transforms.Compose(
            [
                transforms.Resize((int(opt.imsize / 2), int(opt.imsize / 2))),
                transforms.ToTensor(),
            ]
        )

        samples = []
        for d in self.data:
            for i in range(opt.n_layers):
                samples.append(
                    (d["im"], d["masks"][0:i], d["masks"][i], d["classes"][i])
                )  # (im, cur_masks, target_mask, target_label)
        self.samples = samples

    def __getitem__(self, index):
        """
        Returns:
            image: (4=(RGBA), H, W)
            cur_masks: (max_layers, H, W)
            target_mask: (1, H, W)
            # target_class: (1, H, W)
            param: ()
        """
        im, cur_masks, target_mask, target_class = self.samples[index]
        cur_masks = [self.trans_mask(Image.open(p)) for p in cur_masks]
        cur_masks += [
            torch.zeros((1, self.imsize, self.imsize))
            for _ in range(self.n_layers - len(cur_masks))
        ]
        # try:
        #     torch.cat(cur_masks, dim=0)
        # except:
        #     for m in cur_masks:
        #         print(m.shape)

        return (
            self.trans_im(Image.open(im)),
            torch.cat(cur_masks, dim=0),
            self.trans_target_mask(Image.open(target_mask)),
            torch.tensor(target_class).long(),
        )


class ParamDataset(BaseDataset):
    params = ("_rgb", "_a")

    def __init__(self, opt):
        super().__init__(opt)
        samples = []
        for d in self.data:
            for x in zip(d["masks"], d["cats"], d["params"]):
                samples.append((d["im"], *x))
        self.samples = samples

    def __getitem__(self, index):
        """
        Returns:
            image: (4=(RGBA), H, W)
            mask_t: (1, H, W)
            class_t: (1), in int
            param_t: (n_params)
        """
        im, mask, cat, param = self.samples[index]
        p = []
        for k in self.params:
            p += param[k]
        return (
            self.trans_im(Image.open(im)),
            self.trans_mask(Image.open(mask)),
            torch.tensor(cat),
            torch.tensor(p),
        )
