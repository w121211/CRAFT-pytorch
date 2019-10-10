import glob
import json

import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image

import torch
import torchvision.transforms as transforms


class MyDataset(torch.utils.data.Dataset):
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
        self.trans_target_mask = transforms.Compose(
            [
                transforms.Resize((int(opt.imsize / 2), int(opt.imsize / 2))),
                transforms.ToTensor(),
            ]
        )

        # self.blur = GaussianSmoothing(3, 5, 5)
        # self.blur.eval()

        with open("/workspace/CRAFT-pytorch/my-dataset/train/indices.json") as f:
            data = json.load(f)

        samples = []
        for d in data:
            for i in range(opt.n_layers):
                samples.append(
                    (
                        d["image"],
                        d["layers"][0][0:i],
                        d["layers"][0][i],
                        d["layers"][1][i],
                    )
                )  # (im, cur_masks, target_mask, target_class)
        self.samples = samples

    def __getitem__(self, index):
        """
        Returns:
            image: (4=(RGBA), H, W)
            cur_masks: (n_layers, H, W)
            target_mask: (1, H, W)
            # target_class: (1, H, W)
        """
        im, cur_masks, target_mask, target_class = self.samples[index]
        cur_masks = [self.trans_mask(Image.open(p)) for p in cur_masks]
        cur_masks += [
            torch.zeros((1, self.imsize, self.imsize))
            for _ in range(self.n_layers - len(cur_masks))
        ]

        return (
            self.trans_im(Image.open(im)),
            torch.cat(cur_masks, dim=0),
            self.trans_target_mask(Image.open(target_mask)),
        )

    def __len__(self):
        return len(self.samples)
