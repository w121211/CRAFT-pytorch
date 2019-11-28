import glob
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
import cairo
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

CLASSES = ("Null", "Rect", "Photo", "Line")


# define types
Param = Dict[str, Union[int, float, np.ndarray]]


def denorm(key: str, scale: Union[int, float], dtype=np.uint8) -> Callable:
    def fn(param, imsize):
        return tuple((param[key] * scale).astype(dtype))

    return fn


def rgb(param, imsize):
    rgb = (param["_rgb"] * 256).astype(np.uint8)
    return tuple(rgb)


def to_imsize(key):
    def fn(param, imsize):
        p = (param[key] * imsize).astype(np.int16)
        return tuple(p)

    return fn


def bbox(param: dict, imsize: int) -> Tuple[int, int, int, int]:
    _xy = param["_cxy"] - param["_wh"] / 2
    wh = (param["_wh"] * imsize).astype(np.int16)
    xy = (_xy * imsize).astype(np.int16)
    return tuple(np.concatenate((xy, xy + wh)))


class Block(ABC):
    def __init__(self, pspace: OrderedDict, param={}):
        super(Block, self).__init__()
        self._im = None
        self._annotations = None
        self.label = None
        self.param = None
        pspace.update(param)
        self.pspace = pspace

    @abstractmethod
    def sample(self, imsize):
        self.param = None
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
        if bbox is None:
            raise Exception("")
        xy0 = np.array(bbox[:2], dtype=np.float32)
        xy1 = np.array(bbox[2:], dtype=np.float32)
        self.param["_wh"] = (xy1 - xy0) / imsize
        self.param["_cxy"] = ((xy1 + xy0) / 2) / imsize

    def info(self, cmask="Null"):
        # self._annotations = [(type(self).__name__, im)]
        return {
            "cat": CLASSES.index(type(self).__name__),
            "cmask": CLASSES.index(cmask),  # fill cmask
            "bbox": self.param["bbox"],
            "param": self.param,
        }


class Rect(Block):
    def __init__(self, param={}):
        pspace = OrderedDict(
            [
                ("_wh", lambda *a: np.random.normal(0.4, 0.2, 2)),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.random.uniform(0, 1, 1)),
                ("rgb", denorm("_rgb", 256)),
                ("bbox", bbox),
            ]
        )
        super().__init__(pspace, param)

    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.rectangle(self.param["bbox"], fill=self.param["rgb"], outline=None)
        return im, self.info()


class Ellipse(Rect):
    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.ellipse(self.param["bbox"], fill=self.param["rgb"], outline=None)
        return im, self.info()


class Photo(Block):
    def __init__(self, root, param={}):
        # super().__init__(param)
        self.samples = glob.glob(root + "/*.jpg") + glob.glob(root + "/*.png")
        pspace = OrderedDict(
            [
                ("_rgb", lambda *a: np.array([0., 0., 0.])),
                ("_a", lambda *a: np.array([0.])),
                ("_wh", lambda *a: np.random.normal(0.8, 0.2, 2)),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                (
                    "i_photo",
                    lambda *a: np.random.randint(0, len(self.samples), 1)[0],
                ),
                ("wh", to_imsize("_wh")),
                ("cxy", to_imsize("_cxy")),
                ("bbox", bbox),
                ("repeat", lambda *a: False),
            ]
        )
        super().__init__(pspace, param)

    def sample(self, imsize):
        super().sample(imsize)
        cx, cy = self.param["cxy"]
        im = Image.new("RGBA", (imsize, imsize))
        p = Image.open(self.samples[self.param["i_photo"]])

        if self.param["repeat"]:
            for i in range(0, self.param["wh"][0], p.size[0]):
                for j in range(0, self.param["wh"][1], p.size[1]):
                    im.paste(p, (i, j))
                    # print(i, j)
        else:
            p.thumbnail(self.param["wh"])
            im.paste(p, (int(cx - p.width / 2), int(cy - p.height / 2)))

        self.update_param(im.getbbox(), imsize)
        return im, self.info()



class Line(Block):
    fake = Faker()
    fonts = []
    for f in glob.glob("/workspace/mmdetection/my_dataset/fonts_en/**/*.ttf"):
        try:
            _ = ImageFont.truetype(f)
            fonts.append(f)
        except:
            pass
    
    def __init__(self):
        pspace = OrderedDict(
            [
                ("i_font", lambda *a: np.random.randint(0, len(self.fonts), 1)[0]),
                ("textsize", lambda *a: int(np.clip(np.random.normal(5, 3, 1), 1, None)[0])),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.array([1.])),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("rgb", rgb),
                ("cxy", to_imsize("_cxy")),
                # ("a"),
                # ("stroke_w"),
                # ("stoke_rgb"),
            ]
        )
        super().__init__(pspace)

    def sample(self, imsize):
        super().sample(imsize)
        text = self.fake.sentence(nb_words=3, variable_nb_words=True)
        font = ImageFont.truetype(
            self.fonts[self.param["i_font"]], self.param["textsize"]
        )
        w, h = font.getsize(text)
        cx, cy = self.param["cxy"]

        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=self.param["rgb"])

        self.update_param(im.getbbox(), imsize)
        return im, self.info()


class Group(Block):
    def __init__(self, blocks: list, cat=None):
        self.blocks = blocks
        if cat is not None and cat not in CLASSES:
            raise Exception("`cat` must be included in `CLASSES`")
        self.cat = cat  # 判斷是否輸出整個group

    def sample(self, imsize):
        im = Image.new("RGBA", (imsize, imsize))
        info = []
        for bk in self.blocks:
            _im, _info = bk.sample(imsize)
            im.alpha_composite(_im)
            info.append(_info)
        return im, info
        # return zip(*[bk.sample(imsize) for bk in self.blocks])


class CropMask(Block):
    def __init__(self, cmask: Block, base: Rect):
        self.cmask = cmask
        self.base = base

    def sample(self, imsize):
        cim, _ = self.cmask.sample(imsize)
        p = self.cmask.param
        self.base.pspace.update({"_wh": p["_wh"], "_cxy": p["_cxy"]})
        bim, binfo = self.base.sample(imsize)

        im = Image.composite(bim, Image.new("RGBA", (imsize, imsize)), cim)
        binfo.update({"cmask": binfo["cat"]})

        return im, binfo


class Choice(Block):
    def __init__(self, choices=[]):
        self.choices = choices

    def sample(self, imsize):
        return random.choice(self.choices).sample(imsize)


class GradientFill(Rect):
    CLASSES = ("linear", "radial")

    def __init__(self):
        super().__init__()
        psapce = OrderedDict(
            [
                ("i_grad", lambda *a: np.random.randint(0, len(self.CLASSES), 1)[0]),
                ("n_points", lambda *a: np.random.uniform(0, 1, 3)),
                ("_nd_rgb", lambda *a: np.random.uniform(0, 1, 3)),
            ]
        )
        self.pspace.update(psapce)

    def sample(self, imsize):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200)
        cr = cairo.Context(surface)
        cr.scale(200, 200)
        pat = cairo.LinearGradient(0.0, 0.0, 0.0, 1.0)
        pat.add_color_stop_rgba(1, 0, 0, 0, 1)
        pat.add_color_stop_rgba(0, 1, 1, 1, 1)
        cr.rectangle(0, 0, 1, 1)
        cr.set_source(pat)
        cr.fill()

        data = np.ndarray(
            shape=(200, 200, 4), dtype=np.uint8, buffer=surface.get_data()
        )
        im = Image.fromarray(data)
        return im, self.info()

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
        self._annotations = bk.annotations
        self.label = bk.label


class Icon:
    pass
