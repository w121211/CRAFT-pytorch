import argparse
import io
import json
import os
import random
import glob

import numpy as np
from PIL import Image

from mask import create_mask, to_bbox
import blocks as bk


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--save_to", type=str, default="../../my-dataset")

    return parser.parse_args()


class Sampler:
    def __init__(self, blocks: list, opt: argparse.Namespace):
        self.blocks = blocks
        self.imsize = opt.imsize

    def sample(self):
        im = Image.new("RGBA", (self.imsize, self.imsize))
        anns = dict()
        mask_t, cat_t, param_t = [], [], []

        def _save(info):
            mask_t.append(
                create_mask((self.imsize, self.imsize), np.expand_dims(info["bbox"], 0))
            )
            cat_t.append(info["cat"])
            param_t.append(info["param"])

        for bk in self.blocks:
            _im, info = bk.sample(self.imsize)
            im.alpha_composite(_im)
            try:
                for i in info:
                    _save(i)
            except:
                _save(info)

            # for k, v in bk.annotations:
            #     anns.setdefault(k, []).append(v)

        return im, (mask_t, cat_t, param_t)


if __name__ == "__main__":
    """
    Output a json file, eg
        [{
            id: 111,
            im: "path/to/im",
            canvas: ["path/to/canvas1", ...],
            masks: ["path/to/mask1", "path/to/mask2"],
            classes: [0, 2],
            params: [(...), (...)],
        }, {...},, ...]
    """
    opt = get_parameters()
    os.makedirs(os.path.join(opt.save_to, "train"), exist_ok=True)

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
        [bk.Rectangle(p), bk.Photo("/workspace/CoordConv-pytorch/data/facebook", p)]
    )
    text = bk.Choice(
        [
            bk.Line(),
            bk.Group([bk.Line(), bk.Line()]),
            bk.Group([bk.Line(), bk.Line(), bk.Line()]),
        ]
    )
    icon = bk.Rectangle()

    samplers = [
        Sampler([bg, icon, text], opt),
        # Sampler([bg, jpg, text], opt),
        # Sampler([bg, rect, text], opt),
        # Sampler([bg, rect, text], opt),
        # Sampler([bg, jpg, rect, text], opt),
        # Sampler([bg, rect, jpg, text], opt),
        # Sampler([bg, jpg, rect, rect, text], opt),
        # Sampler([bg, rect, jpg, rect, text], opt),
        # Sampler([bg, rect, rect, jpg, text], opt),
        # Sampler([bg, rect, text, rect], opt),
    ]

    info = []
    for i in range(opt.n_samples):
        if i % 100 == 0:
            print(i)
        try:
            im, (mask_t, cat_t, param_t) = random.choice(samplers).sample()
        except:
            continue

        # print(im)
        p_im = os.path.join(opt.save_to, "train", "{}.png".format(i))
        im.save(p_im)

        # print(mask_t, cat_t, params_t)
        masks = []
        # for j, (m, c, p) in enumerate(zip(mask_t, cat_t, params_t)):
        for j, (m, c) in enumerate(zip(mask_t, cat_t)):
            p_mask = os.path.join(
                opt.save_to, "train", "{}_mask_{}_{}.png".format(i, j, c)
            )
            # p_canvas = os.path.join(opt.save_to, "canvas_{}_{}_{}.png".format(i, j, c))

            m.save(p_mask)
            masks.append(os.path.abspath(p_mask))

        info.append(
            {
                "id": i,
                "im": os.path.abspath(p_im),
                "masks": masks,
                "cats": cat_t,
                "params": param_t,
            }
        )

        # count = 0
        # for k, v in anns.items():
        #     for im in v:
        #         im.save(
        #             os.path.join(
        #                 opt.save_to,
        #                 "annotations",
        #                 "{}_crowd_{}_{}.png".format(i, k, count),
        #             ),
        #             "PNG",
        #         )
        #         count += 1

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
            # return json.JSONEncoder.default(self, obj)

    with open(os.path.join(opt.save_to, "train.json"), "w") as f:
        json.dump(info, f, cls=NpEncoder)
