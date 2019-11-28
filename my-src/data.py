import glob
import json

import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image

import torch
import torchvision.transforms as transforms


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


    def __init__(self, json_path, cats: tuple):
        super().__init__(json_path)
        samples = []

        def save(bk, im, mask=None, pre_cat="None", next_cat="None"):
            if bk["cat"] == "Blend":
                _bks = bk["bks"]
                for i in range(len(_bks)):
                    pre_cat = _bks[i - 1]["cat"] if i > 0 else "Blend"
                    next_cat = _bks[i + 1]["cat"] if i < len(_bks) - 1 else "None"
                    save(_bks[i], im, bk["mask"], pre_cat, next_cat)
            elif bk["cat"] == "Crop":
                pass
            else:
                samples.append(
                    (
                        im,
                        mask or bk["mask"],
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
        im, mask, cat, param, pre_cat, next_cat = self.samples[index]

        p = np.array([])
        for k, v in self.param_filter:
            p = np.concatenate((p, param[k] if k in param.keys() else v))

        i_pattern = param["i_sample"] + 1 if self.cats[cat] == "Pattern" else 0
        i_font = param["i_font"] + 1 if self.cats[cat] == "Line" else 0

        return (
            self.trans_im(Image.open(im)),
            self.trans_mask(Image.open(mask)),
            torch.tensor(cat),
            torch.tensor(pre_cat),
            torch.tensor(p).float(),
            [torch.tensor(i) for i in (i_pattern, i_font, next_cat)],
        )
