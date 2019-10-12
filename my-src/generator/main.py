import argparse
import io
import json
import os
import random

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
    def __init__(self, blocks, opt):
        self.blocks = blocks
        self.imsize = opt.imsize

    def sample(self):
        im = Image.new("RGBA", (self.imsize, self.imsize))
        anns = dict()

        im_t, mask_t, label_t, params_t = [], [], [], []
        for bk in self.blocks:
            bk.sample(self.imsize)
            im.alpha_composite(bk.im)
            for k, v in bk.annotations:
                anns.setdefault(k, []).append(v)

            im_t.append(im.copy())
            mask_t.append(
                create_mask(
                    (self.imsize, self.imsize), np.expand_dims(bk.label["box"], 0)
                )
            )
            label_t.append(bk.label["class"])
            # params_t.append(bk.label["params"])
            # layers.append(
            #     (
            #         bk.label["class"],
            #         create_mask(
            #             (self.imsize, self.imsize), np.expand_dims(bk.label["box"], 0)
            #         ),
            #     )
            # )
        return im, anns, (im_t, mask_t, label_t, params_t)


if __name__ == "__main__":
    opt = get_parameters()

    os.makedirs(os.path.join(opt.save_to, "train"), exist_ok=True)

    rect = bk.Rectangle()
    jpg = bk.Photo("/workspace/CoordConv/data/flickr")
    # jpg = bk.Photo("/workspace/CoordConv-pytorch/data/facebook")
    text = bk.Text()
    bg = bk.Background([bk.Rectangle(), bk.Photo("/workspace/CoordConv/data/flickr")])
    # bg = bk.Background(
    #     [bk.Rectangle(), bk.Photo("/workspace/CoordConv-pytorch/data/facebook")]
    # )
    samplers = [
        Sampler([bg, jpg, rect], opt),
        Sampler([bg, jpg, text], opt),
        Sampler([bg, rect, text], opt),
        # Sampler([bg, jpg, rect, text], opt),
        # Sampler([bg, rect, jpg, text], opt),
        # Sampler([bg, jpg, rect, rect, text], opt),
        # Sampler([bg, rect, jpg, rect, text], opt),
        # Sampler([bg, rect, rect, jpg, text], opt),
        # Sampler([bg, rect, text, rect], opt),
    ]

    indices = []
    for i in range(opt.n_samples):
        im, anns, (im_t, mask_t, label_t, params_t) = random.choice(samplers).sample()
        p = os.path.join(opt.save_to, "train", "{}.png".format(i))
        im.save(p)

        masks = []
        for j, (im, mask, label) in enumerate(zip(im_t, mask_t, label_t)):
            p_mask = os.path.join(
                opt.save_to, "train", "{}_mask_{}_{}.png".format(i, j, label)
            )
            # p_canvas = os.path.join(opt.save_to, "canvas_{}_{}_{}.png".format(i, j, c))
            mask.save(p_mask)
            masks.append(os.path.abspath(p_mask))

        indices.append(
            {"id": i, "im": os.path.abspath(p), "masks": masks, "labels": label_t}
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

    with open(os.path.join(opt.save_to, "train.json"), "w") as f:
        json.dump(indices, f)
