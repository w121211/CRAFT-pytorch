"""
Output a json file, eg
    [{
        id: 111,
        im: "path/to/im",
        # canvas: ["path/to/canvas1", ...],
        # masks: ["path/to/mask1", "path/to/mask2"],
        bks: [
            {cat: "...", param: {...}},
            {cat: "Blend", bks: [{cat: "...", param: {...}}, {cat: "...", param: {...}}, ...]},
            {cat: "Crop", bks: [{cat: "...", param: {...}}, {cat: "...", param: {...}}, ...]},
        ]
    }, {...}, {...}, ...]
"""

import argparse
import io
import json
import os
import random
import glob

import numpy as np
from PIL import Image

import blocks as bk
from mask import create_mask


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--save_to", type=str, default="../../my-dataset")
    parser.add_argument("--folder", type=str, default="train")

    return parser.parse_args()


class Sampler:
    def __init__(self, blocks: list, opt: argparse.Namespace):
        self.blocks = blocks
        self.imsize = opt.imsize

    def sample(self):
        im = Image.new("RGBA", (self.imsize, self.imsize))
        bks = []
        for bk in self.blocks:
            # _im, (cat, param, bks) = bk.sample()  # im as annotation
            _im, info = bk.sample(self.imsize)
            im.alpha_composite(_im)
            bks.append(info)
        return im, bks


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.flatten().tolist()
        elif isinstance(obj, Image.Image):
            return None
        return super().default(obj)


if __name__ == "__main__":
    opt = get_parameters()
    os.makedirs(os.path.join(opt.save_to, opt.folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_to, opt.folder, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_to, opt.folder, "masks"), exist_ok=True)

    # rect = bk.Rectangle()
    # jpg = bk.Photo("/workspace/CoordConv/data/flickr")
    # jpg = bk.Photo("/workspace/CoordConv-pytorch/data/facebook")
    # text = bk.Line()

    p = {
        "_a": np.array([1.0]),
        "_wh": np.array([1.0, 1.0]),
        "_cxy": np.array([0.5, 0.5]),
    }
    bg = bk.Choice(
        [bk.Rect(p), bk.Photo("/workspace/CoordConv-pytorch/data/facebook", p)]
        # [bk.Rect(p), bk.Photo("/workspace/CoordConv/data/flickr", p)]
    )
    rect = bk.Rect(p)
    pat = bk.Pattern("/workspace/transparent-textures/patterns", p)

    jpg = bk.Photo("/workspace/CoordConv/data/flickr")
    text = bk.Choice(
        [
            bk.Line(),
            bk.Group([bk.Line(), bk.Line()]),
            bk.Group([bk.Line(), bk.Line(), bk.Line()]),
        ]
    )
    icon = bk.Rect()

    grad = bk.Gradient()
    # crop = bk.CropMask(bk.Group([bk.Line(), bk.Line()]), bk.Gradient())
    crop = bk.CropMask(
        bk.Line({"_rgb": np.array([1.0, 1.0, 1.0]), "_a": np.array([1.0])}),
        bk.Gradient(),
    )
    # crop = bk.CropMask(bk.Line(), bk.Rect())
    pattern = bk.Pattern("/workspace/CRAFT-pytorch/data/patterns")
    blend = bk.Blend([bk.Rect(p), pattern])

    samplers = [
        # Sampler([blend], opt)
        Sampler([bg, grad], opt),
        # Sampler([bg, icon, text], opt),
        # Sampler([bg, jpg, icon, text], opt),
        # Sampler([bg, icon, jpg, text], opt),
        # Sampler([bg, rect, text], opt),
        # Sampler([bg, rect, text], opt),
        # Sampler([bg, jpg, rect, text], opt),
        # Sampler([bg, rect, jpg, text], opt),
        # Sampler([bg, jpg, rect, rect, text], opt),
        # Sampler([bg, rect, jpg, rect, text], opt),
        # Sampler([bg, rect, rect, jpg, text], opt),
        # Sampler([bg, rect, text, rect], opt),
    ]

    meta = {"ns_cats": [len(pattern.samples) + 1]}

    i = 0
    entries = []
    while i < opt.n_samples:
        if i % 100 == 0:
            print(i)

        # im, bks = sample()
        im, bks = random.choice(samplers).sample()
        try:
            im, bks = random.choice(samplers).sample()
        except:
            continue
        im_path = os.path.join(opt.save_to, opt.folder, "images", "{}.png".format(i))
        im.save(im_path)

        for j, bk in enumerate(bks):
            bk["ann"].save(
                os.path.join(
                    opt.save_to,
                    opt.folder,
                    "annotations",
                    "{}_{}_{}.png".format(i, bk["cat"], j),
                )
            )
            mask = create_mask((opt.imsize, opt.imsize), bk["bbox"])
            mask_path = os.path.join(
                opt.save_to, opt.folder, "masks", "{}_{}_{}.png".format(i, bk["cat"], j)
            )
            mask.save(mask_path)
            bk["mask"] = os.path.abspath(mask_path)
        entries.append({"id": i, "im": os.path.abspath(im_path), "bks": bks})
        i += 1

    with open(os.path.join(opt.save_to, opt.folder + ".json"), "w") as f:
        json.dump(dict(meta=meta, entries=entries), f, cls=NpEncoder)
