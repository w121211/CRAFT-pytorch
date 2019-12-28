import copy
import glob
import random
import re
from collections import OrderedDict
from typing import Tuple, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

from .base import Block, to_imsize, denorm
from . import layout


def load_fonts(roots: List[str]) -> List[ImageFont.FreeTypeFont]:
    fonts = []
    for r in roots:
        for f in glob.glob(r):
            try:
                _ = ImageFont.truetype(f)
                fonts.append(f)
            except Exception as e:
                # print(f, e)
                pass
    return fonts


def rand_xy(wh: np.ndarray, imsize: int) -> np.ndarray:
    m = np.clip(np.array([imsize, imsize]) - wh, 0, None)
    x = np.random.randint(0, m[0]+1)
    y = np.random.randint(0, m[1]+1)
    return np.array([x, y])


def rand_text(fake: Faker, lang: int):
    parts = [" ".join(fake.words(np.random.randint(1, 6)))
             for _ in range(np.random.randint(1, 5))]
    for i, p in enumerate(parts):
        if random.random() < 0.2:
            if random.random() < 0.5:
                parts[i] = p.split(" ")
            else:
                parts[i] = [p]
    return parts


fonts_en = load_fonts(["/workspace/post-generator/asset/fonts_en/**/*.ttf"])
fonts_cn = load_fonts(["/workspace/post-generator/asset/fonts_cn/**/*.ttf",
                       "/workspace/post-generator/asset/fonts_cn/**/*.otf"])
fake = Faker(OrderedDict([
    ('en-US', 1),
    ('zh_CN', 2),
    ('zh_TW', 3),
]))


class EmptyBoxError(Exception):
    pass


class Opt:
    def __init__(self, choices=[]):
        self.choices = choices

    def one(self):
        e = random.choice(self.choices)
        if e is None:
            return None
        if callable(e):
            return e()
        if isinstance(e, Opt):
            return e.one()
        if isinstance(e, tuple):
            return self._tobox(e)

    def _tobox(self, e):
        p, children = e
        b = _Box(fonts_en, p)
        for c in children:
            if isinstance(c, tuple):
                c = self._tobox(c)
            elif callable(c):
                c = c()
            elif isinstance(c, Opt):
                c = c.one()
            else:
                raise Exception

            if c is not None:
                b.insert(c)

        if len(b.children) == 0:
            raise EmptyBoxError
        elif len(b.children) == 1 and isinstance(b.children[0], _Box):
            return b.children[0]
        else:
            return b


def _ln(): return _Text(
    " ".join(fake["en-US"].words(np.random.randint(3, 7))), 0, fonts_en)


def _wd(): return _Text(
    " ".join(fake["en-US"].words(np.random.randint(1, 4))), 0, fonts_en)


hkw = Opt([
    ({"i_align": 0}, [_wd, ({"i_align": 0}, [_wd]), _wd]),
    ({"i_align": 0}, [_wd, ({"i_align": 0}, [_wd])]),
    ({"i_align": 0}, [({"i_align": 0}, [_wd]), _wd]),
])
# hln = Opt([None, None, _ln, _ln, hkw])
# hln = Opt([kw])
mln = Opt([
    ({"i_align": 1, "dapart": 0}, [_ln, _ln]),
    ({"i_align": 1, "dapart": 0}, [hkw, _ln]),
    ({"i_align": 1, "dapart": 0}, [_ln, hkw]),
    ({"i_align": 1, "dapart": 0}, [_ln, _ln, _ln]),
    ({"i_align": 1, "dapart": 0}, [_ln, _ln, hkw]),
    ({"i_align": 1, "dapart": 0}, [_ln, hkw, _ln]),
    ({"i_align": 1, "dapart": 0}, [_ln, _ln, hkw]),
    #     (HOR, [vln, vln, hln, hln, hln]),
])


class Textbox(Block):
    fake = {
        0: fake["en-US"],
        1: fake["zh_CN"],
        2: fake["zh_TW"],
    }
    fonts = {
        0: fonts_en,
        1: fonts_cn,
        2: fonts_cn,
    }

    def __init__(self, param={}):
        pspace = OrderedDict(
            [
                # 0: en, 1: cn, 2: tw
                # ("i_lang", lambda *a: random.choice([0, 1, 2])),
                ("i_lang", lambda *a: random.choice([1])),
                ("n_words", lambda *a: np.random.randint(3, 10)),
                ("n_lines", lambda *a: np.random.randint(1, 4)),
            ]
        )
        super().__init__(pspace)

    def sample(self, imsize):
        super().sample(imsize)

        # _text = ["aaa bbb ccc ", ["ddd "]]
        # # root = to_box(
        # #     rand_text(self.fake[self.param["i_lang"]]), self.param["i_lang"])
        # t = rand_text(self.fake[self.param["i_lang"]])
        # print(t)
        # root = to_box(t, self.param["i_lang"])
        root = mln.one()
        im, infos = root.sample(imsize)
        return im, infos


class _Text(layout.Box, Block):
    def __init__(self, t: str, lang: int, fonts: List[ImageFont.FreeTypeFont]):
        super().__init__()
        self.t = t
        self.lang = lang
        self._fonts = fonts
        self._font = None  # temporary store to avoid repeated loading

    def __repr__(self):
        return "<{}>".format(self.t)

    def sample(self, imsize):
        raise NotImplementedError

    def _get_wh(self) -> Tuple[int, int]:
        # TODO: 尚未考慮多行
        self._font = ImageFont.truetype(
            self._fonts[self.param["i_font"]], self.param["textsize"]
        )
        return self._font.getsize(self.t)

    def _render(self, imsize: int):
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)

        tks = re.split("(\s)", self.t) if self.lang == 0 else list(self.t)

        dx, y = tuple(self.xy)
        infos = []
        for tk in tks:
            if len(tk) == 0:
                continue
            draw.text((dx, y), tk, self.param["rgb"], self._font)

            _im = Image.new("RGBA", (imsize, imsize))
            _draw = ImageDraw.Draw(_im)
            _draw.text((dx, y), tk, self.param["rgb"], self._font)
            _bbox = _im.getbbox()
            if _bbox is not None:
                infos.append(self.info(_im, bbox=_bbox, cat="Token"))

            w, _ = self._font.getsize(tk)
            dx += w

        if len(infos) > 1:
            self.update_param(im.getbbox(), imsize)
            infos.append(self.info(im, cat="Box"))

        # draw.text(tuple(self.xy), self.t, self.param["rgb"], self._font)
        self._font = None  # avoid calling this function twice
        return im, infos


class _Box(layout.Box, Block):
    max_param_shift = 3  # max number of parameters to modify

    def __init__(self, fonts, param={}):
        pspace = OrderedDict(
            [
                ("i_font", lambda *a: np.random.randint(0, len(fonts))),
                ("textsize", lambda *a: int(np.clip(np.random.normal(20, 5), 7, None))),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.array([1.0])),
                ("_xy", lambda *a: np.random.uniform(0, 1, 2)),
                ("rgb", denorm("_rgb", 256)),
                ("xy", to_imsize("_xy")),
                # 0 for hor, 1 for ver
                ("i_align", lambda *a: random.choice([0, 1])),
                # ("i_align", lambda *a: i_align),
                (
                    "anchor",
                    lambda *a: random.choice([0.0, 0.5, 1.0]),
                    # lambda *a: random.choice([None]),
                ),  # None for random
                ("dapart", lambda p, *a:
                 int(np.clip(np.random.normal(p["textsize"], p["textsize"] / 4), 0, None))),
            ]
        )
        super().__init__(pspace, param)
        self.param_shift = OrderedDict(
            [
                ("i_font", lambda *a: np.random.randint(0, len(fonts))),
                (
                    "textsize",
                    lambda p:
                    p + int(np.clip(np.random.normal(5, 3), -3, None)),
                ),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                # ("i_align", lambda *a: random.choice([0, 1])),
            ]
        )
        self.after_param_shift = OrderedDict(
            [
                ("dapart", lambda p, *a:
                 int(np.clip(np.random.normal(p["textsize"], p["textsize"] / 4), 0, None))),
                ("rgb", denorm("_rgb", 256)),
            ]
        )
        self.after_param_shift.update(param)

    def __repr__(self) -> str:
        return "<Box: {}>".format(str(self.children))

    def sample(self, imsize: int) -> Tuple[Image.Image, List[dict]]:
        self._sample_param(imsize)
        wh = self._sample_align()
        self.wh = np.array(wh, dtype=np.int32)

        print(self.wh)

        # TODO: 確保在Image裡面
        # self.xy = np.array(self.param["xy"], dtype=np.int32)
        self.xy = self.param["xy"] = rand_xy(self.wh, imsize)
        self.set_xy()

        return self._render(imsize)

    def _sample_param(self, imsize, parent=None):
        if parent is None:
            super().sample(imsize)
        else:
            for k in random.sample(
                self.param_shift.keys(), np.random.randint(1, self.max_param_shift + 1)
                # self.param_shift.keys(), np.random.randint(1, 2)
            ):
                p = copy.deepcopy(parent.param)
                p[k] = self.param_shift[k](parent.param[k])

            for k, v in self.after_param_shift.items():
                if callable(v):
                    p[k] = v(p, imsize)
                else:
                    p[k] = v
            self.param = p

        for c in self.children:
            if isinstance(c, _Box):
                c._sample_param(imsize, self)
            elif isinstance(c, _Text):
                c.param = self.param
            else:
                print(c)
                raise Exception("Unsupported type")

    def _sample_align(self):
        m = {0: layout.HOR, 1: layout.VER}

        # init each child's (w, h) in order to align
        for c in self.children:
            if isinstance(c, _Box):
                wh = c._sample_align()
            elif isinstance(c, _Text):
                wh = c._get_wh()
            else:
                raise Exception("Unknown type")
            c.wh = np.array(wh, dtype=np.int32)
            c.param["wh"] = wh

        return layout.align(
            self.children, by=m[self.param["i_align"]],
            anchor=self.param["anchor"], dapart=self.param["dapart"],
        )

    def _render(self, imsize):
        im = Image.new("RGBA", (imsize, imsize))
        # draw = ImageDraw.Draw(im)
        # bbox = np.concatenate((self.xy, self.xy + self.wh))
        # draw.rectangle(tuple(bbox), fill=None, outline=(200, 0, 0))

        infos = []
        for c in self.children:
            _im, _infos = c._render(imsize)
            im.alpha_composite(_im)
            infos += _infos

        self.update_param(im.getbbox(), imsize)
        infos.append(self.info(im, cat="Box"))

        return im, infos
