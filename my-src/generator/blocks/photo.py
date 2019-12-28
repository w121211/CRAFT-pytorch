import glob
import random
import copy
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

from .base import Block, to_imsize, bbox

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
                # iaa.Invert(0.05, per_channel=True),  # invert color channels
                iaa.Add(
                    (-10, 10), per_channel=0.5
                ),  # change brightness of images (by -10 to 10 of original value)
                # iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
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
                # iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                # iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            ],
        ),
    ],
    random_order=True,
)


class Photo(Block):
    def __init__(self, root, param={}, cat=None, aug=True):
        # self.samples = glob.glob(root + "/*.jpg") + glob.glob(root + "/*.png")
        self.samples = glob.glob(root)
        pspace = OrderedDict(
            [
                # ("_wh", lambda *a: np.random.normal(0.8, 0.2, 2)),
                ("_wh", lambda *a: np.clip(np.random.normal(0.6, 0.2, 2), 0.2, 1)),
                # ("_wh", lambda *a: np.random.normal(0.6, 0.2, 2)),
                ("_cxy", lambda *a: np.random.uniform(0, 1, 2)),
                ("i_sample", lambda *a: np.random.randint(0, len(self.samples), 1)[0]),
                ("wh", to_imsize("_wh")),
                ("cxy", to_imsize("_cxy")),
                ("bbox", bbox),
                # ("repeat", lambda *a: False),
                ("_xy", lambda *a: np.random.uniform(0, 0.7, 2)),  # use for photo group
                ("xy", to_imsize("_xy")),
            ]
        )
        super().__init__(pspace, param, cat)

        self._photo = None

    def render(self, imsize):
        # p = imageio.imread(self.samples[self.param["i_sample"]])
        # p = photo_seq.augment_image(p)
        # p = Image.fromarray(p).convert("RGBA")
        # p.thumbnail(self.param["wh"])
        p = self._photo.resize(self.param["wh"], Image.BICUBIC)
        im = Image.new("RGBA", (imsize, imsize))
        im.paste(p, tuple(self.param["xy"]))
        self.update_param(im.getbbox(), imsize)
        return im, self.info(im)

    def sample(self, imsize):
        super().sample(imsize)
        cx, cy = self.param["cxy"]

        # p = imageio.imread(self.samples[self.param["i_sample"]])
        # p = photo_seq.augment_image(p)
        # p = Image.fromarray(p).convert("RGBA")
        p = Image.open(self.samples[self.param["i_sample"]]).convert("RGBA")

        self._photo = copy.deepcopy(p)

        p.thumbnail(self.param["wh"])

        im = Image.new("RGBA", (imsize, imsize))
        im.paste(p, (int(cx - p.width / 2), int(cy - p.height / 2)))

        self.update_param(im.getbbox(), imsize)
        return im, self.info(im)


class PhotoGroup(Block):
    def __init__(self, photo):
        self.photo = photo
        pspace = OrderedDict(
            [
                # ("n", lambda *a: np.random.randint(1, 5, 2)),  # (n_hor, n_ver)
                ("n", lambda *a: np.random.randint(3, 5)),
                ("dir", lambda *a: bool(random.getrandbits(1))),
            ]
        )
        super().__init__(pspace)

    def _cat_hor(self, photos):
        min_h = min(p.param["wh"][1] for p in photos)
        x, y = photos[0].param["xy"]
        dx = 0
        for p in photos:
            w, h = p.param["wh"]
            p.param["wh"] = np.array((w * min_h / h, min_h), dtype=np.int32)
            p.param["xy"] = np.array((x + dx, y), dtype=np.int32)
            dx += p.param["wh"][0]

    def _cat_ver(self, photos):
        min_w = min(p.param["wh"][0] for p in photos)
        x, y = photos[0].param["xy"]
        dy = 0
        for p in photos:
            w, h = p.param["wh"]
            p.param["wh"] = np.array((min_w, h * min_w / w), dtype=np.int32)
            p.param["xy"] = np.array((x, y + dy), dtype=np.int32)
            dy += p.param["wh"][1]

    def sample(self, imsize):
        super().sample(imsize)
        photos = []
        for _ in range(self.param["n"]):
            p = copy.deepcopy(self.photo)
            p.sample(imsize)
            photos.append(p)

        if self.param["dir"]:
            self._cat_hor(photos)
        else:
            self._cat_ver(photos)
        # self._cat_hor(photos)

        # ims, infos = list(zip(p.render(imsize) for p in photos))
        ims, infos = [], []
        for p in photos:
            # im, info = p.render(imsize)
            # ims.append(im)
            # infos.append(info)
            try:
                im, info = p.render(imsize)
                ims.append(im)
                infos.append(info)
            except:
                pass
        return ims, infos
