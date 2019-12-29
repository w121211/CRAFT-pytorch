import glob
import random
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Union, Callable, Dict, Type

import numpy as np
import cairo
from PIL import Image, ImageDraw
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

from .layout import rand_box

# define types
Param = Dict[str, Union[int, float, np.ndarray]]

photo_seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        # iaa.Flipud(0.2),  # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        # iaa.Sometimes(
        #     iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))
        # ),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.SomeOf(
            (1, 3),
            [
                iaa.SimplexNoiseAlpha(
                    iaa.OneOf(
                        [
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0.5, 1.0), direction=(0.0, 1.0)
                            ),
                        ]
                    )
                ),
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),  # add gaussian noise to images
                iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf(
                    [
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0)),
                        ),
                    ]
                ),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            ],
        ),
    ],
    random_order=True,
)

icon_seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.1),  # vertically flip 20% of all images
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
        iaa.Sometimes(
            0.5, iaa.Affine(rotate=(-45, 45))  # rotate by -45 to +45 degrees
        ),
        iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
    ],
    random_order=True,
)


def denorm(key: str, scale: Union[int, float], dtype=np.int32) -> Callable:
    def fn(param, imsize):
        if param[key] is None:
            return None
        return tuple((param[key] * scale).astype(dtype))

    return fn


def to_imsize(key):
    def fn(param, imsize):
        p = (param[key] * imsize).astype(np.int32)
        return tuple(p)

    return fn


def bbox(param: dict, imsize: int) -> Tuple[int, int, int, int]:
    _xy = param["_cxy"] - param["_wh"] / 2
    wh = (param["_wh"] * imsize).astype(np.int32)
    xy = (_xy * imsize).astype(np.int32)
    return tuple(np.concatenate((xy, xy + wh)))


class Block(ABC):
    def __init__(self, pspace=OrderedDict(), param={}, cat=None):
        super().__init__()
        self._im = None
        self.label = None
        self.param = None
        pspace.update(param)
        self.pspace = pspace
        self.cat = cat or type(self).__name__

    @abstractmethod
    def sample(self, imsize, *args, **kwargs):
        p = dict()
        for k, v in self.pspace.items():
            if callable(v):
                p[k] = v(p, imsize)
            else:
                p[k] = v
        self.param = p

    def update_param(self, bbox: Tuple[int, int, int, int], imsize: int):
        """Given bbox, update param's `wh`, `cxy`. Used for undetermined shape"""
        self.param["bbox"] = bbox
        # if bbox is None:
        #     raise Exception()
        if bbox is not None:
            xy0 = np.array(bbox[:2], dtype=np.float32)
            xy1 = np.array(bbox[2:], dtype=np.float32)
            self.param["_wh"] = (xy1 - xy0) / imsize
            self.param["_cxy"] = ((xy1 + xy0) / 2) / imsize

    def info(self, ann: Image.Image, bbox=None, cat=None, bk_infos=None):
        return {
            "ann": ann,
            "cat": cat or self.cat,
            "bbox": bbox or self.param["bbox"],
            "param": self.param,
            "bks": bk_infos,
        }


class Rect(Block):
    def __init__(self, param={}):
        pspace = OrderedDict(
            [
                ("_wh", lambda *a: np.clip(np.random.normal(0.4, 0.2, 2), 0.03, 1)),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.random.uniform(0.3, 1, 1)),
                ("rgb", denorm("_rgb", 256)),
                ("a", denorm("_a", 256)),
                ("bbox", bbox),  # (x1, y1, x2, y2)
            ]
        )
        super().__init__(pspace, param)

    def sample(self, imsize, bbox=None, *args, **kwargs):
        super().sample(imsize)
        bbox = bbox or self.param["bbox"]

        rgba = self.param["rgb"] + self.param["a"]
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox, fill=rgba, outline=None)

        return im, self.info(im, bbox)


class Ellipse(Rect):
    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.ellipse(self.param["bbox"], fill=self.param["rgb"], outline=None)
        return im, self.info(im)


class Grad(Rect):
    CLASSES = ("linear", "radial")

    def __init__(self, param={}):
        super().__init__(param)

        def _wh(p, *a):
            _wh = np.clip(np.random.normal(0.4, 0.2, 2), 0.05, 1)
            _wh = np.clip(_wh, 0, 2 - 2 * p["_cxy"])
            _wh = np.clip(_wh, 0, 2 * p["_cxy"])
            return _wh

        def _xy(*a):
            if bool(random.getrandbits(1)):
                return np.array([1.0, np.random.uniform(0, 1)])
            else:
                return np.array([np.random.uniform(0, 1), 1.0])

        pspace = OrderedDict(
            [
                ("_cxy", lambda *a: np.random.uniform(0.05, 0.95, 2)),
                ("_wh", _wh),
                # ("i_cat", lambda *a: np.random.randint(0, len(self.CLASSES), 1)[0]),
                # ("_stops", lambda *a: np.random.randint(2, 3, 1)[0]),  # ()
                ("_stop_xy2", _xy),
                (
                    "_stop_rgba",
                    lambda *a: np.random.uniform(0, 1, (2, 4)),
                ),  # (n_stops, rgba)
            ]
        )
        tmp = pspace.copy()  # 為了更新OrderedDict的順序
        tmp.update(self.pspace)
        tmp.update(pspace)
        tmp.update(param)  # 外部的param直接寫入
        self.pspace = tmp

    def sample(self, imsize):
        super().sample(imsize)
        x1, y1, x2, y2 = self.param["bbox"]
        w, h = x2 - x1, y2 - y1
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        cr = cairo.Context(surface)
        cr.scale(w, h)
        pat = cairo.LinearGradient(0.0, 0.0, *self.param["_stop_xy2"])
        pat.add_color_stop_rgba(0.0, *self.param["_stop_rgba"][0])
        pat.add_color_stop_rgba(1.0, *self.param["_stop_rgba"][1])
        # pat.add_color_stop_rgba(0, 1, 1, 1, 1)
        # pat.add_color_stop_rgba(1, 0, 0, 0, 1)
        cr.rectangle(0, 0, 1, 1)
        cr.set_source(pat)
        cr.fill()
        data = np.ndarray(shape=(h, w, 4), dtype=np.uint8,
                          buffer=surface.get_data())
        p = Image.fromarray(data)

        im = Image.new("RGBA", (imsize, imsize))
        im.paste(p, (x1, y1))
        return im, self.info(im)


class Pattern(Rect):
    def __init__(self, root, param={}):
        super().__init__(param)
        self.samples = glob.glob(root + "/*.jpg") + glob.glob(root + "/*.png")
        pspace = OrderedDict(
            [
                ("i_sample", lambda *a: np.random.randint(0,
                                                          len(self.samples), 1)[0]),
                # ("i_cat", lambda *a: np.random.randint(0, len(self.CLASSES), 1)[0]),
            ]
        )
        self.pspace.update(pspace)

    def sample(self, imsize):
        rect, _ = super().sample(imsize)

        fill = Image.new("RGBA", (imsize, imsize))
        p = Image.open(self.samples[self.param["i_sample"]]).convert("RGBA")
        for i in range(0, imsize, p.size[0]):
            for j in range(0, imsize, p.size[1]):
                fill.paste(p, (i, j))

        blend = Image.alpha_composite(rect, fill)
        im = Image.new("RGBA", (imsize, imsize))
        im.paste(blend, mask=rect)
        # im.paste(blend, mask=rect.split()[-1])

        return im, self.info(im)


class Icon(Rect):
    PATTERN = ("tile", "rand")

    def __init__(self, root, param={}):
        super().__init__(param)
        self.samples = glob.glob(root + "/*.png")
        pspace = OrderedDict(
            [
                ("_wh", lambda *a: np.clip(np.random.normal(0.2, 0.1, 2), 0.05, 0.7)),
                ("i_sample", lambda *a: np.random.randint(0, len(self.samples))),
                ("transpose", None),  # (skew, rotate), stretch is defined by wh
            ]
        )
        self.pspace.update(pspace)

    def sample(self, imsize):
        super().sample(imsize)
        x1, y1, x2, y2 = self.param["bbox"]
        w, h = x2 - x1, y2 - y1
        rgba = self.param["rgb"] + self.param["a"]

        p = imageio.imread(self.samples[self.param["i_sample"]])
        p = icon_seq.augment_image(p)
        p = Image.fromarray(p[:, :, 3])  # alpha layer only
        p.thumbnail((w, h))

        crop = Image.new("L", (imsize, imsize))
        crop.paste(p, (x1, y1))
        fill = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(fill)
        draw.rectangle((0, 0, imsize, imsize), fill=rgba, outline=None)
        im = Image.composite(fill, Image.new(
            "RGBA", (imsize, imsize)), mask=crop)

        return im, self.info(im)


class Choice(Block):
    def __init__(self, choices: List[Block] = []):
        assert len(choices) >= 1
        self.choices = choices

    def sample(self, imsize):
        return random.choice(self.choices).sample(imsize)


class Filter(Rect):
    def __init__(self, mask):
        super().__init__()
        self.param_space = [
            ("i_bk", lambda *acc: np.random.randint(0, len(choices), 1)[0]),
            ("_wh", lambda *args: np.array([1.0, 1.0])),
            ("_cxy", lambda *args: np.array([0.5, 0.5])),
            # ("_rgb", lambda *args: np.random.uniform(0, 1, 3)),
            # ("rgb", rgb),
            # ("cxy", to_imsize("_cxy")),
            # ("wh", to_imsize("_wh")),
        ]
        for bk in blocks:
            bk._override_param_space(self.param_space)
        self.choices = choices

    def sample(self, imsize):
        super().sample(imsize)
        bk = self.choices[self.param["i_bk"]]
        bk.sample(imsize)
        self._im = bk.im
        self.label = bk.label


class CropMask(Block):
    def __init__(self, mask: Block, fill: Rect):
        self.mask = mask
        self.fill = fill

    def sample(self, imsize):
        m_im, m_info = self.mask.sample(imsize)
        p = self.mask.param
        self.fill.pspace.update({"_wh": p["_wh"], "_cxy": p["_cxy"]})
        f_im, f_info = self.fill.sample(imsize)

        im = Image.composite(f_im, Image.new("RGBA", (imsize, imsize)), m_im)
        f_info.update({"cmask": m_info["cat"]})

        # return im, f_info
        return (
            im,
            self.info(
                m_im,
                cat="{}&{}".format(type(self.mask).__name__,
                                   type(self.fill).__name__),
            ),
        )


class Blend(Block):
    def __init__(self, blocks: List[Block], cat: str, crop=False):
        super().__init__(cat=cat)
        assert len(blocks) >= 2
        self.blocks = blocks
        self.crop = crop

    def sample(self, imsize):
        crop = self.blocks[0]
        c_im, c_info = crop.sample(imsize)

        infos = [c_info]
        f_im = Image.new("RGBA", (imsize, imsize))
        for b in self.blocks[1:]:
            b.pspace.update(
                {"_wh": crop.param["_wh"], "_cxy": crop.param["_cxy"]})
            _im, _info = b.sample(imsize)
            f_im.alpha_composite(_im)
            infos.append(_info)

        im = Image.composite(f_im, Image.new("RGBA", (imsize, imsize)), c_im)
        return im, self.info(im, bbox=crop.param["bbox"], bk_infos=infos)


class Copies(Block):
    def __init__(self, bk: Block, min: int, max: int, lock_params=tuple()):
        pspace = OrderedDict(
            [("n_copies", lambda *a: np.random.randint(min, max))])
        super().__init__(pspace)
        self.bk = bk
        self.lock_params = lock_params

    def sample(self, imsize):
        super().sample(imsize)
        im, _ = self.bk.sample(imsize)
        ims, infos = [], []
        for _ in range(self.param["n_copies"]):
            cp = copy.deepcopy(self.bk)  # use copy avoid pollute
            for p in self.lock_params:
                cp.pspace.update({p: self.bk.param[p]})
            im, info = cp.sample(imsize)
            ims.append(im)
            infos.append(info)

        return ims, infos


class BoxLayoutGroup(Block):
    def __init__(self, blocks: List[Type[Rect]]):
        def _wh(p, *a):
            _wh = np.clip(np.random.normal(0.6, 0.2, 2), 0.3, 1)
            _wh = np.clip(_wh, 0, 2 - 2 * p["_cxy"])
            _wh = np.clip(_wh, 0, 2 * p["_cxy"])
            return _wh

        pspace = OrderedDict(
            [
                ("_cxy", lambda *a: np.random.uniform(0.15, 0.85, 2)),
                ("_wh", _wh),
                ("bbox", bbox),  # (x1, y1, x2, y2)
            ]
        )
        super().__init__(pspace)
        self.blocks = blocks

    def sample(self, imsize: int):
        super().sample(imsize)
        x0, y0, x1, y1 = self.param["bbox"]
        root = rand_box(np.array([x1 - x0, y1 - y0]))
        root.xy = np.array([x0, y0], dtype=np.int32)
        root.set_xy()

        ims, infos = [], []
        for b in root.leafs():
            cp = copy.deepcopy(random.choice(self.blocks))
            _im, _info = cp.sample(imsize, tuple(b.bbox()))
            ims.append(_im)
            infos.append(_info)

        return ims, infos
