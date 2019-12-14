import glob
import json

import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image

import torch
import torchvision.transforms as transforms

from .generator.mask import create_mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, json_path):
        self.trans_im = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
            ]
        )
        self.trans_mask = transforms.Compose([transforms.ToTensor()])

        with open(json_path) as f:
            self.data = json.load(f)
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplementedError()


class ParamDataset(BaseDataset):
    param_filter = (
        ("_rgb", np.zeros(3)),
        ("_a", np.zeros(1)),
        ("_textsize", np.zeros(1)),
        ("_stop_xy2", np.zeros(2)),
        ("_stop_rgba", np.zeros(4 * 2)),
    )
    p_dim = 15
    cat_filter = (("Pattern", "i_sample"), ("Icon", "i_sample"), ("Line", "i_font"))

    def __init__(self, json_path, cats: tuple, imsize: int):
        super().__init__(json_path)
        samples = []

        def save(bk, im, bbox=None, pre_cat="None", next_cat="None"):
            if "Blend" in bk["cat"]:
                _bks = bk["bks"]
                for i in range(len(_bks)):
                    pre_cat = _bks[i - 1]["cat"] if i > 0 else bk["cat"]
                    next_cat = _bks[i + 1]["cat"] if i < len(_bks) - 1 else "None"
                    save(_bks[i], im, bk["param"]["bbox"], pre_cat, next_cat)
            elif bk["cat"] == "Crop":
                pass
            else:
                samples.append(
                    (
                        im,
                        bbox,
                        cats.index(bk["cat"]),
                        bk["param"],
                        cats.index(pre_cat),
                        cats.index(next_cat),
                    )
                )

        for d in self.data["entries"]:
            for bk in d["bks"]:
                save(bk, d["im"])

        # from pprint import pprint
        # pprint(samples)
        self.imsize = imsize
        self.meta = self.data["meta"]
        self.samples = samples
        self.cats = cats

    def __getitem__(self, index):
        """
        Returns:
            image: (4=(RGBA), H, W)
            mask: (1, H, W)
            class: (1), in int
            param: (n_params)
        """
        im, bbox, cat, param, pre_cat, next_cat = self.samples[index]

        # 隨機調整bbox
        xy = np.array(bbox[:2]) + np.random.randint(-5, 5, 2)  # 不需clip，轉成mask後自動會clip
        wh = np.clip(
            (np.array(bbox[:2]) - np.array(bbox[-2:])) + np.random.randint(-5, 5, 2),
            5,
            10000000,
        )
        bbox = np.concatenate([xy, xy + wh])
        mask = create_mask((self.imsize, self.imsize), bbox)

        p = np.array([])
        for k, v in self.param_filter:
            p = np.concatenate((p, param[k] if k in param.keys() else v))

        
        i_pattern = param["i_sample"] + 1 if self.cats[cat] == "Pattern" else 0
        i_font = param["i_font"] + 1 if self.cats[cat] == "Line" else 0

        return (
            self.trans_im(Image.open(im)),
            self.trans_mask(mask),
            torch.tensor(cat),
            torch.tensor(pre_cat),
            torch.tensor(p).float(),
            [torch.tensor(i) for i in (i_pattern, i_font, next_cat)],
        )
