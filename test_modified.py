"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description="CRAFT Text Detection")
parser.add_argument(
    "--trained_model",
    default="weights/craft_mlt_25k.pth",
    type=str,
    help="pretrained model",
)
parser.add_argument(
    "--text_threshold", default=0.7, type=float, help="text confidence threshold"
)
parser.add_argument("--low_text", default=0.4, type=float, help="text low-bound score")
parser.add_argument(
    "--link_threshold", default=0.4, type=float, help="link confidence threshold"
)
parser.add_argument(
    "--cuda", default=True, type=str2bool, help="Use cuda to train model"
)
parser.add_argument(
    "--canvas_size", default=1280, type=int, help="image size for inference"
)
parser.add_argument(
    "--mag_ratio", default=1.5, type=float, help="image magnification ratio"
)
parser.add_argument(
    "--show_time", default=False, action="store_true", help="show processing time"
)
parser.add_argument(
    "--test_folder", default="/data/", type=str, help="folder path to input images"
)

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = "./result/"
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def test_net(net, image, text_threshold, link_threshold, low_text, cuda):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image,
        args.canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=args.mag_ratio,
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        torch.cuda.empty_cache()
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].unsqueeze(0).cpu()
    # score_link = y[0, :, :, 1].unsqueeze(0)

    score_text = torch.clamp(score_text, 0, 1)
    score_text = score_text * (score_text > 0.01).float()

    
    return score_text


if __name__ == "__main__":
    # load net
    net = CRAFT()  # initialize

    if args.cuda:
        net = net.cuda()
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    print("Loading weights from checkpoint (" + args.trained_model + ")")
    # net.load_state_dict(torch.load(args.trained_model))
    state_dict = torch.load(args.trained_model, map_location="cpu")

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove 'module.' of dataparallel
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    transform = transforms.ToPILImage()
    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print(
            "Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path),
            end="\r",
        )
        image = imgproc.loadImage(image_path)

        score_text = test_net(
            net,
            image,
            args.text_threshold,
            args.link_threshold,
            args.low_text,
            args.cuda,
        )

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask = transform(score_text.cpu())
        mask.save(result_folder + "/" + filename + "_mask.png", "PNG")

    print("elapsed time : {}s".format(time.time() - t))
