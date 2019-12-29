import argparse
import datetime
import json
import os
import re
import fnmatch
import numpy as np
from PIL import Image
from pycococreatortools import pycococreatortools

from main import CATEGORIES

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="train")
opt = parser.parse_args()

# PROJECT_ROOT = os.path.join(os.getcwd(), os.pardir)
PROJECT_ROOT = os.path.join(os.getcwd())
ROOT_DIR = os.path.join(PROJECT_ROOT, "../../my-dataset/" + opt.folder)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
    }
]


def png_to_jpg(root, files):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    print(files)

    for path in files:
        im = Image.open(path)
        im.convert("RGB").save(path.replace(".png", ".jpg"))


def filter_for_jpeg(root, files):
    file_types = ["*.jpeg", "*.jpg", "*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ["*.png"]
    file_types = r"|".join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + ".*"
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [
        f
        for f in files
        if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])
    ]
    return files


def png_to_binary_mask(fpath: str) -> np.array:
    im = Image.open(fpath)
    bw = Image.new("1", im.size)
    pixels = im.load()
    pixels_bw = bw.load()

    for i in range(im.size[0]):
        for j in range(im.size[1]):
            if pixels[i, j][3] > 128:
                pixels_bw[i, j] = 1
            else:
                pixels_bw[i, j] = 0

    mask = np.asarray(bw).astype(np.uint8)
    return mask


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 1
    segmentation_id = 1

    # filter for jpeg images
    for root, _, files in os.walk(IMAGE_DIR):
        # png_to_jpg(root, files)
        image_files = filter_for_jpeg(root, files)

        # go through each image
        for image_filename in image_files:
            print(image_filename)

            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size
            )
            coco_output["images"].append(image_info)

            # filter for associated png annotations
            for root, _, files in os.walk(ANNOTATION_DIR):
                annotation_files = filter_for_annotations(root, files, image_filename)

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    # TODO: naive bug fix，限定於cat是按照複雜度排列，仍有可能出錯
                    class_id = [
                        x["id"] for x in CATEGORIES if x["name"] in annotation_filename
                    ][-1]

                    category_info = {
                        "id": class_id,
                        # "is_crowd": "crowd" in image_filename,
                        "is_crowd": True,  # 強制讓segmentation用png->binary mask
                    }
                    binary_mask = png_to_binary_mask(annotation_filename)
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id,
                        image_id,
                        category_info,
                        binary_mask,
                        image.size,
                        # tolerance=10,
                        tolerance=2,
                    )

                    # print(annotation_info)
                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1

    with open("{}/annotation.json".format(ROOT_DIR), "w") as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
