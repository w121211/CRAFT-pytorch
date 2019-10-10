from __future__ import absolute_import

import glob
import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gym import spaces
from faker import Faker


class Block(ABC):
    def __init__(self):
        super(Block, self).__init__()
        self.param_space = []
        self._im = None
        self._annotations = None
        self.param = None
        self.label = None

    @property
    def im(self):
        if self._im is None:
            raise Exception()
        else:
            return self._im

    @property
    def annotations(self):
        if self._annotations is None:
            raise Exception()
        else:
            return self._annotations

    def _reset(self):
        self._im = None
        self.param = None
        self.label = None

    def _override_param_space(self, dst_space):
        dst = dict()
        for k, fn in dst_space:
            dst[k] = fn
        self.param_space = [
            (k, dst[k]) if k in dst.keys() else (k, fn) for k, fn in self.param_space
        ]

    @abstractmethod
    def sample(self, imsize):
        self._reset()
        param = dict()
        for k, fn in self.param_space:
            param[k] = fn(param, imsize)
        self.param = param


def rgb(param, imsize):
    rgb = (param["_rgb"] * 256).astype(np.uint8)
    return tuple(rgb)


def to_imsize(key):
    def fn(param, imsize):
        p = (param[key] * imsize).astype(np.int16)
        return tuple(p)

    return fn


def wh(param, imsize):
    wh = (param["_wh"] * imsize).astype(np.int16)
    return tuple(wh)


def box(param, imsize):
    _xy = param["_cxy"] - param["_wh"] / 2
    wh = (param["_wh"] * imsize).astype(np.int16)
    xy = (_xy * imsize).astype(np.int16)
    box = np.concatenate((xy, xy + wh))
    return tuple(box)


class Rectangle(Block):
    def __init__(self):
        super(Rectangle, self).__init__()
        self.param_space = [
            ("_wh", lambda *args: np.random.normal(0.4, 0.2, 2)),
            ("_cxy", lambda *args: np.random.uniform(0, 1, 2)),
            ("_rgb", lambda *args: np.random.uniform(0, 1, 3)),
            ("rgb", rgb),
            ("box", box),
        ]

    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.rectangle(self.param["box"], fill=self.param["rgb"], outline=None)
        self._im = im
        self._annotations = [(type(self).__name__, im)]
        self.label = {"class": "rectangle", "box": self.param["box"]}


class Photo(Block):
    def __init__(self, root):
        super(Photo, self).__init__()
        self.samples = glob.glob(root + "/*.jpg")
        self.param_space = [
            ("_wh", lambda *args: np.random.normal(0.8, 0.2, 2)),
            ("_cxy", lambda *args: np.random.uniform(0, 1, 2)),
            ("_rgb", lambda *args: np.random.uniform(0, 1, 3)),
            ("rgb", rgb),
            ("wh", to_imsize("_wh")),
            ("cxy", to_imsize("_cxy")),
            ("box", box),
            ("idx", lambda *args: np.random.randint(0, len(self.samples), 1)[0]),
        ]

    def sample(self, imsize):
        super().sample(imsize)
        cx, cy = self.param["cxy"]
        im = Image.new("RGBA", (imsize, imsize))
        p = Image.open(self.samples[self.param["idx"]])
        p.thumbnail(self.param["wh"])
        im.paste(p, (int(cx - p.width / 2), int(cy - p.height / 2)))
        self._im = im
        self._annotations = [(type(self).__name__, im)]
        self.label = {"class": "photo", "box": im.getbbox()}


class Text(Block):
    def __init__(self):
        super(Text, self).__init__()
        self.fake = Faker()

        fonts = []
        for f in glob.glob("/workspace/mmdetection/my_dataset/fonts_en/**/*.ttf"):
            try:
                _ = ImageFont.truetype(f)
                fonts.append(f)
            except:
                pass
        self.fonts = fonts
        self.param_space = [
            ("i_font", lambda *args: np.random.randint(0, len(self.fonts), 1)[0]),
            ("textsize", lambda *args: int(np.random.normal(12, 3, 1)[0])),
            ("_cxy", lambda *args: np.random.uniform(0, 1, 2)),
            ("_rgb", lambda *args: np.random.uniform(0, 1, 3)),
            ("rgb", rgb),
            ("cxy", to_imsize("_cxy")),
        ]

    def sample(self, imsize):
        super().sample(imsize)

        text = self.fake.sentence(nb_words=7, variable_nb_words=True)
        font = ImageFont.truetype(
            self.fonts[self.param["i_font"]], self.param["textsize"]
        )

        w, h = font.getsize(text)
        cx, cy = self.param["cxy"]

        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=self.param["rgb"])
        self._im = im
        self._annotations = [(type(self).__name__, im)]
        self.label = {"class": "text", "box": im.getbbox()}


class Icon:
    pass


class Effect:
    pass


class Component(Block):
    pass


class Background(Block):
    def __init__(self, choices=[]):
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
        for bk in choices:
            bk._override_param_space(self.param_space)
        self.choices = choices

    def sample(self, imsize):
        super().sample(imsize)
        bk = self.choices[self.param["i_bk"]]
        bk.sample(imsize)
        self._im = bk.im
        self._annotations = bk.annotations
        self.label = bk.label
