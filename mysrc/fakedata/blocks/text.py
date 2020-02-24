import copy
import glob
import random
import re
from collections import OrderedDict
from typing import Tuple, List, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

from .base import Block, to_imsize, denorm, bbox
from .lorem import cn_lorem
from . import layout


def load_fonts(roots: List[str]) -> List[ImageFont.FreeTypeFont]:
    fonts = []
    for r in roots:
        for f in glob.glob(r):
            try:
                font = ImageFont.truetype(f)
                fonts.append(f)
            except Exception as e:
                pass
    return fonts


fonts_en = load_fonts(["/workspace/post-generator/asset/fonts_en/**/*.ttf"])
# fonts_cn = load_fonts(["/workspace/post-generator/asset/fonts_cn/**/*.ttf",
#                        "/workspace/post-generator/asset/fonts_cn/**/*.otf"])
# fonts_cn = None
fonts_cn = load_fonts(["/workspace/post-generator/asset/fonts_cn/**/*.otf"])

fake = Faker(OrderedDict([
    ('en-US', 1),
    ('zh_CN', 2),
    ('zh_TW', 3),
]))


def rand_xy(wh: np.ndarray, imsize: int) -> np.ndarray:
    m = np.clip(np.array([imsize, imsize]) - wh, 0, None)
    x = np.random.randint(0, m[0] + 1)
    y = np.random.randint(0, m[1] + 1)
    return np.array([x, y])


def rand_lines(i_lang, max_words, max_lines) -> List[str]:
    f, _, _ = {
        0: (fake["en-US"], "This is a sample text", 16),
        1: (fake["zh_CN"], "这是一个很简单的中文测试", 16),
        2: (fake["zh_TW"], "這是一個很簡單的中文測試", 16)
    }[i_lang]

    # n_lines = np.radnom.randint(1, max_lines + 1)
    n_lines = np.clip(np.random.normal(1, 2), 1, max_lines).astype(np.int32)
    # lines = [f.text(np.random.randint(5, max_words + 1))[:-1]
    #          for _ in range(n_lines)]
    lines = [cn_lorem(np.random.randint(5, max_words + 1))
             for _ in range(n_lines)]

    return lines


def rand_text_by_wh(box_wh: np.ndarray, i_lang: int, font_path: str) -> str:
    f, t, s = {
        0: (fake["en-US"], "This is a sample text", 16),
        1: (fake["zh_CN"], "这是一个很简单的中文测试", 16),
        2: (fake["zh_TW"], "這是一個很簡單的中文測試", 16)
    }[i_lang]

    bw, bh = box_wh
    font = ImageFont.truetype(font_path, s)
    w, h = font.getsize(t)

    try:
        text = f.text(max_nb_chars=int(len(t) * bw / w))[:-1]
        w, h = font.getsize(text)
        fontsize = int(s * bw / w)
    except:
        text = ""
        fontsize = 0

    return text, fontsize


def randbox():
    # return layout.rand_box(np.array([500, 500])).clone(_Box, fonts_cn, 1)
    return layout.Box().clone(_Box, fonts_cn, 1)

# def rand_text(fake: Faker, lang: int):
#     parts = [" ".join(fake.words(np.random.randint(1, 6)))
#              for _ in range(np.random.randint(1, 5))]
#     for i, p in enumerate(parts):
#         if random.random() < 0.2:
#             if random.random() < 0.5:
#                 parts[i] = p.split(" ")
#             else:
#                 parts[i] = [p]
#     return parts


class EmptyBoxError(Exception):
    pass


class EmptyLineError(Exception):
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


# def _ln(): return _Text(
#     " ".join(fake["en-US"].words(np.random.randint(3, 7))), 0, fonts_en)


# def _wd(): return _Text(
#     " ".join(fake["en-US"].words(np.random.randint(1, 4))), 0, fonts_en)


# hkw = Opt([
#     ({"i_align": 0}, [_wd, ({"i_align": 0}, [_wd]), _wd]),
#     ({"i_align": 0}, [_wd, ({"i_align": 0}, [_wd])]),
#     ({"i_align": 0}, [({"i_align": 0}, [_wd]), _wd]),
# ])
# hln = Opt([None, None, _ln, _ln, hkw])
# hln = Opt([kw])
# mln = Opt([
#     ({"i_align": 1, "dapart": 0}, [_ln, _ln]),
#     ({"i_align": 1, "dapart": 0}, [hkw, _ln]),
#     ({"i_align": 1, "dapart": 0}, [_ln, hkw]),
#     ({"i_align": 1, "dapart": 0}, [_ln, _ln, _ln]),
#     ({"i_align": 1, "dapart": 0}, [_ln, _ln, hkw]),
#     ({"i_align": 1, "dapart": 0}, [_ln, hkw, _ln]),
#     ({"i_align": 1, "dapart": 0}, [_ln, _ln, hkw]),
#     #     (HOR, [vln, vln, hln, hln, hln]),
# ])


class Textbox(Block):
    fake = {
        0: fake["en-US"],
        1: fake["zh_CN"],
        2: fake["zh_TW"],
    }
    fonts = {
        0: fonts_en,
        # 1: fonts_cn,
        # 2: fonts_cn,
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


class BoxLayoutGroupText(Block):
    def __init__(self):
        def _wh(p, *a):
            _wh = np.clip(np.random.normal(0.7, 0.1, 2), 0.5, 1)
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

    def sample(self, imsize: int):
        super().sample(imsize)
        x0, y0, x1, y1 = self.param["bbox"]
        root = layout.rand_box(np.array([x1 - x0, y1 - y0]))
        root.xy = np.array([x0, y0], dtype=np.int32)
        root.set_xy()

        ims, infos = [], []
        for b in root.leafs():
            cp = copy.deepcopy(random.choice(self.blocks))
            _im, _info = cp.sample(imsize, tuple(b.bbox()))
            ims.append(_im)
            infos.append(_info)

        return ims, infos


class _Line(layout.Box, Block):
    def __init__(self, t: str):
        super().__init__()
        self.t = t

    def __repr__(self):
        return "<{}>".format(self.t)

    def sample(self, imsize):
        raise NotImplementedError

    def _render(self, imsize: int, font: ImageFont, param: dict):
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.text(self.xy, self.t, param["rgb"], font)

        bbox = im.getbbox()
        if bbox is None:
            raise EmptyLineError

        return im, [self.info(im, cat="Line", bbox=bbox)]
        # return im, []

    # def _render(self, imsize: int):
    #     im = Image.new("RGBA", (imsize, imsize))
    #     draw = ImageDraw.Draw(im)
    #     tks = re.split(r"(\s)", self.t) if self.lang == 0 else list(self.t)

    #     dx, y = tuple(self.xy)
    #     infos = []
    #     for tk in tks:
    #         if len(tk) == 0:
    #             continue
    #         draw.text((dx, y), tk, self.param["rgb"], self._font)

    #         _im = Image.new("RGBA", (imsize, imsize))
    #         _draw = ImageDraw.Draw(_im)
    #         _draw.text((dx, y), tk, self.param["rgb"], self._font)
    #         _bbox = _im.getbbox()
    #         # if _bbox is not None:
    #         #     infos.append(self.info(_im, bbox=_bbox, cat="Token"))

    #         w, _ = self._font.getsize(tk)
    #         dx += w

    #     if len(infos) > 1:
    #         self.update_param(im.getbbox(), imsize)
    #         infos.append(self.info(im, cat="Box"))

    #     # draw.text(tuple(self.xy), self.t, self.param["rgb"], self._font)
    #     self._font = None  # avoid calling this function twice
    #     return im, infos


class _Box(layout.Box, Block):
    max_param_shift = 3  # max number of parameters to modify

    def __init__(self, fonts, i_lang, param={}):
        self._fonts = fonts
        self._font = None  # save load time

        pspace = OrderedDict(
            [
                # ("i_lang", lambda *a: random.choice([1])),
                ("i_lang", i_lang),
                ("i_font", lambda *a: np.random.randint(0, len(fonts))),
                ("fontsize", lambda *a: int(np.clip(np.random.normal(20, 5), 7, None))),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.array([1.0])),
                ("_xy", lambda *a: np.random.uniform(0, 0.5, 2)),
                ("rgb", denorm("_rgb", 256)),
                ("xy", to_imsize("_xy")),
                # 0 for hor, 1 for ver
                ("i_align", lambda *a: random.choice([0, 1])),
                # ("i_align", lambda *a: 1),
                (
                    "anchor",
                    lambda *a: random.choice([0.0, 0.5, 1.0]),
                    # lambda *a: random.choice([None]),
                ),  # None for random
                ("dapart", lambda p, *a:
                 int(np.clip(np.random.normal(p["fontsize"] / 2, p["fontsize"] / 4), 0, None))),
                # ("dapart", lambda p, *a: 0)
            ]
        )
        super().__init__(pspace, param)

        # self.param_shift = OrderedDict(
        #     [
        #         ("i_font", lambda *a: np.random.randint(0, len(fonts))),
        #         (
        #             "textsize",
        #             lambda p:
        #             p + int(np.clip(np.random.normal(5, 3), -3, None)),
        #         ),
        #         ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
        #         # ("i_align", lambda *a: random.choice([0, 1])),
        #     ]
        # )
        # self.after_param_shift = OrderedDict(
        #     [
        #         ("dapart", lambda p, *a:
        #          int(np.clip(np.random.normal(p["textsize"], p["textsize"] / 4), 0, None))),
        #         ("rgb", denorm("_rgb", 256)),
        #     ]
        # )
        # self.after_param_shift.update(param)

    def __repr__(self) -> str:
        return "<_Box({}, {}): {}>".format(self.rxy, self.wh, str(self.children))

    def sample(self, imsize: int) -> Tuple[Image.Image, List[dict]]:
        self = randbox()
        self._sample_param(imsize)

        # sample text
        for b in self.leafs():
            # text, fontsize = rand_text_by_wh(
            #     b.wh, self.param["i_lang"], self._fonts[b.param["i_font"]])
            # if len(text) > 0:
            #     t = _Text(text, self._fonts)
            #     b.param["fontize"] = fontsize
            #     b.insert(t)
            b.param["i_align"] = 1
            b._font = ImageFont.truetype(
                self._fonts[b.param["i_font"]], b.param["fontsize"])

            for t in rand_lines(self.param["i_lang"], max_words=15, max_lines=3):
                b.insert(_Line(t))

        wh = self._sample_align()
        self.wh = np.array(wh, dtype=np.int32)

        # TODO: 確保在Image裡面
        # self.xy = np.array(self.param["xy"], dtype=np.int32)
        self.xy = self.param["xy"] = rand_xy(self.wh, imsize)
        self.set_xy()
        # self.xy = np.zeros(2)
        # self.set_xy()

        return self._render(imsize)

    def _sample_param(self, imsize):
        super().sample(imsize)
        self.param["xy"] = self.xy
        self.param["wh"] = self.wh
        for c in self.children:
            c._sample_param(imsize)

    def _render(self, imsize, *args, **kwargs):
        im = Image.new("RGBA", (imsize, imsize))

        # (for debug) draw bbox
        # if self.xy is not None:
        #     draw = ImageDraw.Draw(im)
        #     bbox = np.concatenate((self.xy, self.xy + self.wh))
        #     draw.rectangle(tuple(bbox), fill=None, outline=(200, 0, 0))

        infos = []
        for c in self.children:
            try:
                if isinstance(c, _Line):
                    font = ImageFont.truetype(
                        self._fonts[self.param["i_font"]], self.param["fontsize"])
                    _im, _infos = c._render(imsize, font, self.param)
                else:
                    _im, _infos = c._render(imsize)

                im.alpha_composite(_im)
                infos += _infos
            except (EmptyLineError, EmptyBoxError):
                pass

        im = im.rotate(np.random.randint(-60, 60))

        bbox = im.getbbox()
        if bbox is None:
            raise EmptyBoxError
        # infos.append(self.info(im, bbox=bbox, cat="Box"))

        return im, infos

    def _sample_align(self):
        m = {0: layout.HOR, 1: layout.VER}

        # init each child's (w, h) in order to align
        for c in self.children:
            if isinstance(c, _Box):
                wh = c._sample_align()
                c.param["wh"] = wh
            elif isinstance(c, _Line):
                wh = self._font.getsize(c.t)
            else:
                raise Exception("Unknown type")
            c.wh = np.array(wh, dtype=np.int32)

        return layout.align(
            self.children, by=m[self.param["i_align"]],
            anchor=self.param["anchor"], dapart=self.param["dapart"],
        )

    def _sample_shift_param(self, imsize, parent=None):
        """@Deprecated"""
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
            else:
                print(c)
                raise Exception("Unsupported type")


# class _FixedBox(_Box):
#     max_param_shift = 3  # max number of parameters to modify
#     base_text = "This is a sample text"
#     base_size = 16

#     def __init__(self, fonts, xy, wh, param={}):
#         param.update(dict(xy=xy, wh=wh))
#         pspace = OrderedDict(
#             [
#                 ("i_font", lambda *a: np.random.randint(0, len(fonts))),
#                 ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
#                 ("_a", lambda *a: np.array([1.0])),
#                 ("rgb", denorm("_rgb", 256)),
#                 # 0 for hor, 1 for ver
#                 ("i_align", lambda *a: random.choice([0, 1])),
#                 (
#                     "anchor",
#                     lambda *a: random.choice([0.0, 0.5, 1.0]),
#                     # lambda *a: random.choice([None]),
#                 ),  # None for random
#                 # ("dapart", lambda p, *a:
#                 #  int(np.clip(np.random.normal(p["textsize"], p["textsize"] / 4), 0, None))),
#             ]
#         )
#         super().__init__(pspace, param)
#         self.param_shift = OrderedDict(
#             [
#                 ("i_font", lambda *a: np.random.randint(0, len(fonts))),
#                 ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
#             ]
#         )
#         self.after_param_shift = OrderedDict(
#             [
#                 # ("dapart", lambda p, *a:
#                 #  int(np.clip(np.random.normal(p["textsize"], p["textsize"] / 4), 0, None))),
#                 ("rgb", denorm("_rgb", 256)),
#             ]
#         )
#         self.after_param_shift.update(param)
#         self._fonts = fonts
#         self.xy = xy
#         self.wh = wh

#     def sample(self, imsize: int) -> Tuple[Image.Image, List[dict]]:
#         self._sample_param(imsize)
#         for b in self.leafs():
#             text, fontsize = rand_text_by_wh(
#                 self.param["wh"], self._fonts[self.param["i_font"]], self.base_text, self.base_size)
#             t = _Text(text, 0, self._fonts)
#             b.param["textsize"] = fontsize
#             # t.xy, t.wh = b.xy, b.wh
#             # print(b.xy)
#             # print(t.xy)
#             b.insert(t)
#         return self._render(imsize)
