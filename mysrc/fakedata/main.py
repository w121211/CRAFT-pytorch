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
import copy
from typing import List

import numpy as np
from PIL import Image

import blocks as bk
from mask import create_mask

CATEGORIES = [
    {"id": 0, "name": "Null", "supercategory": "Block"},
    {"id": 1, "name": "Rect", "supercategory": "Block"},
    {"id": 2, "name": "Grad", "supercategory": "Block"},
    {"id": 3, "name": "Icon", "supercategory": "Block"},
    {"id": 4, "name": "Pattern", "supercategory": "Block"},
    {"id": 5, "name": "Line", "supercategory": "Block"},
    {"id": 6, "name": "Blend", "supercategory": "Block"},
    {"id": 7, "name": "Blend_Rect", "supercategory": "Block"},
    {"id": 8, "name": "Blend_Icon", "supercategory": "Block"},
    {"id": 9, "name": "Photo_Rect", "supercategory": "Block"},
    {"id": 10, "name": "Photo_Trans", "supercategory": "Block"},
    {"id": 11, "name": "Box", "supercategory": "Block"},
    {"id": 12, "name": "Token", "supercategory": "Block"},
]
PARAMS_TO_PREDICT = (
    ("_rgb", np.zeros(3)),
    ("_a", np.zeros(1)),
    ("_textsize", np.zeros(1)),
    ("_stop_xy2", np.zeros(2)),
    ("_stop_rgba", np.zeros(4 * 2)),
)
PARAM_CATS_TO_PREDICT = (
    ("Pattern", "i_sample"),
    ("Icon", "i_sample"),
    ("Line", "i_font"),
)


def decode_segmap(seg: Image.Image, n_classes: int) -> Image.Image:
    """For visualize scence segmentaion annotation"""
    seg = np.array(seg)
    r, g, b = seg.copy(), seg.copy(), seg.copy()
    for i in range(0, n_classes):
        r[seg == i] = 20 * (i % 3)
        g[seg == i] = i * 10
        b[seg == i] = 0
    rgb = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return Image.fromarray(rgb)


class Sampler:
    cats = {c["name"]: c["id"] for c in CATEGORIES}

    def __init__(self, blocks: list, opt: argparse.Namespace):
        self.blocks = blocks
        self.imsize = opt.imsize

    def scene_seg(self, infos: List[dict]) -> Image.Image:
        """Create a scence annotation, returnm a single channel image"""
        seg = np.zeros((self.imsize, self.imsize), dtype=np.uint8)  # label

        cat_ignore = 0
        for info in infos:
            cat = self.cats[info["cat"]]

            im = info["ann"]
            a = im.split()[-1]  # alpha channel
            a = np.array(a)

            # cat-label
            m = a > 240
            seg = seg * np.invert(m) + np.uint8(cat) * m

            # ignore-label
            m = np.logical_and(20 < a, a < 240)
            seg = seg * np.invert(m) + np.uint8(cat_ignore) * m

        return Image.fromarray(seg)

    def sample(self):
        im = Image.new("RGBA", (self.imsize, self.imsize))
        infos = []
        for bk in self.blocks:
            # _im, (cat, param, bks) = bk.sample()  # im as annotation
            _im, _info = bk.sample(self.imsize)
            try:
                for a in _im:
                    im.alpha_composite(a)
            except:
                im.alpha_composite(_im)

            if isinstance(_info, list):
                infos += _info
            else:
                infos.append(_info)

            # if isinstance(_im, list):
            #     for a, b in zip(_im, _info):
            #         im.alpha_composite(a)
            #         bks.append(b)
            # else:
            #     im.alpha_composite(_im)
            #     bks.append(_info)

        return im, infos, self.scene_seg(infos)


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


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--save_to", type=str, default="../../mydataset")
    parser.add_argument("--folder", type=str, default="train")

    return parser.parse_args()


if __name__ == "__main__":
    opt = get_parameters()
    os.makedirs(os.path.join(opt.save_to, opt.folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(
        opt.save_to, opt.folder, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_to, opt.folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(
        opt.save_to, opt.folder, "segmentations"), exist_ok=True)
    os.makedirs(os.path.join(
        opt.save_to, opt.folder, "segmentations_visualize"), exist_ok=True)

    p = {
        "_a": np.array([1.0]),
        "_wh": np.array([1.0, 1.0]),
        "_cxy": np.array([0.5, 0.5]),
    }
    bg = bk.Choice(
        [
            bk.Rect(p),
            bk.Grad(p),
            # bk.Photo("/workspace/CoordConv/data/flickr/*.jpg", p, cat="Photo_Rect"),
            bk.Photo(
                "/workspace/CRAFT-pytorch/data/crawled_fb/model/*.jpg", p, cat="Photo_Rect"
            ),
            bk.Pattern("/workspace/CRAFT-pytorch/data/patterns", p),
        ]
    )

    rect = bk.Rect()
    # pat = bk.Pattern("/workspace/transparent-textures/patterns")
    pat = bk.Pattern("/workspace/CRAFT-pytorch/data/patterns", p)

    # pat_grad= bk.Blend([pat, bk.Gradient()], "Blend_Icon", crop=False)
    bw_grad = bk.Grad({"_stop_rgba": np.array([[0, 0, 0, 0], [1, 1, 1, 1]])})
    bw = bk.Blend([rect, bw_grad], "Blend_Rect")

    # line = bk.Line()
    # lines = bk.Copies(line, 3, 5)

    photo = bk.Choice(
        [
            # bk.Photo("/workspace/CoordConv/data/flickr", p, cat="Photo_Rect"),
            bk.Photo(
                "/workspace/CoordConv/data/freepng/freepngs/freepngs - complete collection/People/**/*.png",
                cat="Photo_Trans",
            ),
            # bk.Photo("..../transparent", cat="Photo_Trans"),
        ]
    )

    _icon = bk.Icon("/workspace/CRAFT-pytorch/data/icons")
    icon = bk.Choice(
        [
            _icon,
            bk.Blend([_icon, bk.Grad()], "Blend_Icon", crop=True),
            bk.Blend([_icon, bk.Rect()], "Blend_Icon", crop=True),
            bk.Blend([_icon, copy.deepcopy(pat)], "Blend_Icon", crop=True),
        ]
    )
    icons = bk.Copies(icon, 1, 5)

    pg = bk.Choice(
        [
            bk.PhotoGroup(bk.Photo(
                "/workspace/CRAFT-pytorch/data/crawled_fb/model/*.jpg", cat="Photo_Rect")),
            # bk.PhotoGroup(bk.Photo("/workspace/CoordConv/data/flickr/*.jpg", cat="Photo_Rect"))
        ]
    )
    # tb = bk.Textbox()
    # tb = bk.randbox()
    blg = bk.BoxLayoutGroup([
        bk.Photo(
            "/workspace/CRAFT-pytorch/data/crawled_fb/model/*.jpg", cat="Photo_Rect"
        ),
        bk.Rect()
    ])

    samplers = [
        Sampler([bg, bk.randbox(), bk.randbox(), bk.randbox(),
                 bk.randbox(), bk.randbox()], opt)
        # Sampler([bg, icons, lines], opt),
        # Sampler([bg, lines, icons], opt),
        # Sampler([bg, pg, icons, lines], opt),
        # Sampler([bg, pg, lines, icon], opt),
        # Sampler([bg, photo, lines, icon], opt),
        # Sampler([bg, rect, icon, text], opt),
        # Sampler([bg, rect, text, icon], opt),
    ]
    meta = {
        "params_to_predict": PARAMS_TO_PREDICT,
        "param_cats_to_predict": PARAM_CATS_TO_PREDICT,
        # "ns_cats": (len(pat.samples) + 1, len(_icon.samples) + 1, len(line.fonts) + 1),
    }

    i = 0
    entries = []
    while i < opt.n_samples:
        if i % 100 == 0:
            print(i)

        # im, infos, seg = random.choice(samplers).sample()
        try:
            im, infos, seg = random.choice(samplers).sample()
        except:
            continue
        im_path = os.path.join(
            opt.save_to, opt.folder, "images", "{}.png".format(i))
        im.save(im_path)
        seg_path = os.path.join(
            opt.save_to, opt.folder, "segmentations", "{}.png".format(i))
        seg.save(seg_path)
        seg_path = os.path.join(
            opt.save_to, opt.folder, "segmentations_visualize", "{}.png".format(i))
        seg = decode_segmap(seg, len(CATEGORIES))
        seg.save(seg_path)

        for j, info in enumerate(infos):
            info["ann"].save(
                os.path.join(
                    opt.save_to,
                    opt.folder,
                    "annotations",
                    "{}_{}_{}.png".format(i, info["cat"], j),
                )
            )

            # mask = create_mask((opt.imsize, opt.imsize), bk["bbox"])
            # mask_path = os.path.join(
            #     opt.save_to, opt.folder, "masks", "{}_{}_{}.png".format(i, bk["cat"], j)
            # )
            # mask.save(mask_path)
            # bk["mask"] = os.path.abspath(mask_path)

        entries.append({"id": i, "im": os.path.abspath(im_path), "bks": infos})
        i += 1

    with open(os.path.join(opt.save_to, opt.folder + ".json"), "w") as f:
        json.dump(dict(meta=meta, entries=entries), f, cls=NpEncoder)
