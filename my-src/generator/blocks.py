import glob
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Tuple, Union, Callable, Dict

import numpy as np
import cairo
from PIL import Image, ImageDraw, ImageFont
from faker import Faker

CLASSES = ("None", "Rect", "Gradient", "Pattern", "Photo", "Line", "Blend")


# define types
Param = Dict[str, Union[int, float, np.ndarray]]


def denorm(key: str, scale: Union[int, float], dtype=np.int16) -> Callable:
    def fn(param, imsize):
        if param[key] is None:
            return None
        return tuple((param[key] * scale).astype(dtype))

    return fn


def to_imsize(key):
    def fn(param, imsize):
        p = (param[key] * imsize).astype(np.int16)
        return tuple(p)

    return fn


# def clip(key, scope):
def clip(param, imsize):

    param["_wh"]
    pass


def bbox(param: dict, imsize: int) -> Tuple[int, int, int, int]:
    _xy = param["_cxy"] - param["_wh"] / 2
    wh = (param["_wh"] * imsize).astype(np.int16)
    xy = (_xy * imsize).astype(np.int16)
    return tuple(np.concatenate((xy, xy + wh)))


class Block(ABC):
    def __init__(self, pspace=OrderedDict(), param={}):
        super().__init__()
        self._im = None
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

    def info(self, ann: Image.Image, bbox=None, cat=None, bk_infos=None):
        return {
            "ann": ann,
            "cat": type(self).__name__ if cat is None else cat,
            "bbox": self.param["bbox"] if bbox is None else bbox,
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
                ("_a", lambda *a: np.random.uniform(0, 1, 1)),
                ("rgb", denorm("_rgb", 256)),
                ("bbox", bbox),  # (x1, y1, x2, y2)
            ]
        )
        super().__init__(pspace, param)

    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.rectangle(self.param["bbox"], fill=self.param["rgb"], outline=None)
        return im, self.info(im)


class Ellipse(Rect):
    def sample(self, imsize):
        super().sample(imsize)
        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.ellipse(self.param["bbox"], fill=self.param["rgb"], outline=None)
        return im, self.info(im)


class Gradient(Rect):
    CLASSES = ("linear", "radial")

    def __init__(self):
        super().__init__()

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
        tmp = pspace.copy()
        tmp.update(self.pspace)
        tmp.update(pspace)
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
        data = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=surface.get_data())
        p = Image.fromarray(data)

        im = Image.new("RGBA", (imsize, imsize))
        im.paste(p, (x1, y1))
        return im, self.info(im)


class Pattern(Rect):
    CLASSES = ("tile",)

    def __init__(self, root, param={}):
        super().__init__(param)
        self.samples = glob.glob(root + "/*.jpg") + glob.glob(root + "/*.png")
        pspace = OrderedDict(
            [
                ("_rgb", lambda *a: np.array([0.0, 0.0, 0.0])),
                ("_a", lambda *a: np.array([0.0])),
                ("i_sample", lambda *a: np.random.randint(0, len(self.samples), 1)[0]),
                ("i_cat", lambda *a: np.random.randint(0, len(self.CLASSES), 1)[0]),
            ]
        )
        self.pspace.update(pspace)

    def sample(self, imsize):
        super().sample(imsize)

        x1, y1, x2, y2 = self.param["bbox"]
        w, h = x2 - x1, y2 - y1

        _im = Image.new("RGBA", (w, h))
        p = Image.open(self.samples[self.param["i_sample"]])
        p = p.convert("RGBA")
        if self.CLASSES[self.param["i_cat"]] == "tile":
            for i in range(0, w, p.size[0]):
                for j in range(0, h, p.size[1]):
                    _im.paste(p, (i, j))

        im = Image.new("RGBA", (imsize, imsize))
        im.paste(_im, (x1, y1))
        return im, self.info(im)


class Photo(Block):
    def __init__(self, root, param={}):
        # super().__init__(param)
        self.samples = glob.glob(root + "/*.jpg") + glob.glob(root + "/*.png")
        pspace = OrderedDict(
            [
                ("_wh", lambda *a: np.random.normal(0.8, 0.2, 2)),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("i_photo", lambda *a: np.random.randint(0, len(self.samples), 1)[0]),
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
        return im, self.info(im)


class Line(Block):
    fake = Faker()
    fonts = []
    for f in glob.glob("/workspace/mmdetection/my_dataset/fonts_en/**/*.ttf"):
        try:
            _ = ImageFont.truetype(f)
            fonts.append(f)
        except:
            pass

    def __init__(self, param={}):
        pspace = OrderedDict(
            [
                ("i_font", lambda *a: np.random.randint(0, len(self.fonts), 1)[0]),
                (
                    "textsize",
                    lambda *a: int(np.clip(np.random.normal(14, 5, 1), 5, None)[0]),
                ),
                ("_rgb", lambda *a: np.random.uniform(0, 1, 3)),
                ("_a", lambda *a: np.array([1.0])),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("rgb", denorm("_rgb", 256)),
                ("cxy", to_imsize("_cxy")),
                # ("a"),
                # ("stroke_w"),
                # ("stoke_rgb"),
            ]
        )
        super().__init__(pspace, param)

    def sample(self, imsize):
        super().sample(imsize)

        text = self.fake.sentence(nb_words=5, variable_nb_words=True)
        font = ImageFont.truetype(
            self.fonts[self.param["i_font"]], self.param["textsize"]
        )
        w, h = font.getsize(text)
        cx, cy = self.param["cxy"]

        im = Image.new("RGBA", (imsize, imsize))
        draw = ImageDraw.Draw(im)
        draw.text((cx - w / 2, cy - h / 2), text, font=font, fill=self.param["rgb"])

        self.update_param(im.getbbox(), imsize)
        return im, self.info(im)


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
                cat="{}&{}".format(type(self.mask).__name__, type(self.fill).__name__),
            ),
        )


class Choice(Block):
    def __init__(self, choices=[]):
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


class Blend(Block):
    def __init__(self, blocks: List[Block]):
        super().__init__()
        self.blocks = blocks

    def sample(self, imsize):
        base = self.blocks[0]
        im, _info = base.sample(imsize)

        infos = [_info]
        for b in self.blocks[1:]:
            b.pspace.update({"_wh": base.param["_wh"], "_cxy": base.param["_cxy"]})
            _im, _info = b.sample(imsize)
            im.alpha_composite(_im)
            infos.append(_info)

        return im, self.info(im, bbox=base.param["bbox"], bk_infos=infos)


class Icon:
    pass
