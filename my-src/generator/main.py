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
    parser.add_argument("--save_to", type=str, default="../../my-dataset/train")

    return parser.parse_args()


class Sampler:
    def __init__(self, blocks, opt):
        self.blocks = blocks
        self.imsize = opt.imsize

    def sample(self):
        im = Image.new("RGBA", (self.imsize, self.imsize))
        anns = dict()

        im_t, mask_t, class_t, params_t = [], [], [], []
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
            class_t.append(bk.label["class"])
            params_t.append(bk.label["params"])
            
            # labels.append(bk.label)
            # print(bk.label)
            # mask, bw = create_mask(
            #     (self.imsize, self.imsize), np.expand_dims(to_bbox(bk.label["box"]), 0)
            # )

            # layers.append(
            #     (
            #         bk.label["class"],
            #         create_mask(
            #             (self.imsize, self.imsize), np.expand_dims(bk.label["box"], 0)
            #         ),
            #     )
            # )
        return im, anns, [im_t, mask_t, class_t, params_t]


if __name__ == "__main__":
    opt = get_parameters()

    os.makedirs(os.path.join(opt.save_to, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_to, "layers"), exist_ok=True)

    rect = bk.Rectangle()
    # jpg = bk.Photo("/workspace/CoordConv/data/flickr")
    jpg = bk.Photo("/workspace/CoordConv-pytorch/data/facebook")
    text = bk.Text()
    # bg = bk.Background([bk.Rectangle(), bk.Photo("/workspace/CoordConv/data/flickr")])
    bg = bk.Background(
        [bk.Rectangle(), bk.Photo("/workspace/CoordConv-pytorch/data/facebook")]
    )
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
        im, anns, layers = random.choice(samplers).sample()
        p = os.path.join(opt.save_to, "images", "{}.png".format(i))
        im.save(p)

        l = [[], []]
        for j, (c, mask) in enumerate(layers):
            _p = os.path.join(opt.save_to, "layers", "{}_{}_{}.png".format(i, j, c))
            mask.save(_p)
            l[0].append(os.path.abspath(_p))
            l[1].append(c)
        indices.append({"id": i, "image": os.path.abspath(p), "layers": l})

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

    with open(os.path.join(opt.save_to, "indices.json"), "w") as f:
        json.dump(indices, f)
