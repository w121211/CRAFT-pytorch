import numpy as np
import cv2
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

__all__ = ["create_mask", "to_bbox"]

sigma = 10
spread = 3
extent = int(spread * sigma)
heatmap = np.zeros([2 * extent, 2 * extent], dtype=np.float32)
for i in range(2 * extent):
    for j in range(2 * extent):
        heatmap[i, j] = (
            1
            / 2
            / np.pi
            / (sigma ** 2)
            * np.exp(
                -1
                / 2
                * ((i - spread * sigma - 0.5) ** 2 + (j - spread * sigma - 0.5) ** 2)
                / (sigma ** 2)
            )
        )
heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)


def create_mask(imsize: tuple, bbox: tuple) -> Image:
    """
    Args:
        imsize: (w, h)
        bboxes: (x0, y0, x1, y1)
    """
    mask = Image.new("L", imsize)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(bbox, fill=255)
    return mask


def create_n_mask(imsize: tuple, bboxes: np.ndarray) -> Image:
    """
    Args:
        imsize: (w, h)
        bboxes: (n_bboxes, 4=(x0, y0, x1, y1))
    """
    mask = Image.new("L", imsize)
    draw = ImageDraw.Draw(mask)
    for bbox in bboxes:
        draw.rectangle(tuple(bbox), fill=255)
    return mask


def create_heat_mask(im_size, bboxes: np.array):
    """
    Args:
        im_size: (h, w)
        bboxes: (n_bboxes, 4, 2), clockwise points of a rectangle
    """
    mask = np.zeros(im_size, dtype=np.uint8)
    for bbox in bboxes:
        print(bbox)
        mask = _add_bbox(mask, bbox)
    return mask / 255.0, np.float32(mask != 0)
    # return mask / 255.0


def _add_bbox(mask: np.array, bbox: np.array):
    if not Polygon(bbox.reshape([4, 2]).astype(np.int32)).is_valid:
        return mask

    top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    if top_left[1] > mask.shape[0] or top_left[0] > mask.shape[1]:
        # This means there is some bug in the character bbox
        # Will have to look into more depth to understand this
        return mask
    bbox -= top_left[None, :]
    warped = _warp(heatmap.copy(), bbox.astype(np.float32))

    start_row = max(top_left[1], 0) - top_left[1]
    start_col = max(top_left[0], 0) - top_left[0]
    end_row = min(top_left[1] + warped.shape[0], mask.shape[0])
    end_col = min(top_left[0] + warped.shape[1], mask.shape[1])

    mask[max(top_left[1], 0) : end_row, max(top_left[0], 0) : end_col] += warped[
        start_row : end_row - top_left[1], start_col : end_col - top_left[0]
    ]
    return mask


def _warp(image: np.array, points: np.array):
    """
    Args:
        image: (H, W)
        points: [[x0, y0], [x1, y1], ...], clockwise points of a rectangle
    """
    max_x, max_y = (
        np.max(points[:, 0]).astype(np.int32),
        np.max(points[:, 1]).astype(np.int32),
    )

    src = np.array(
        [
            [0, 0],
            [image.shape[1] - 1, 0],
            [image.shape[1] - 1, image.shape[0] - 1],
            [0, image.shape[0] - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(src, points)
    warped = cv2.warpPerspective(image, M, (max_x, max_y))

    return warped


def to_bbox(xy0xy1):
    x0, y0, x1, y1 = xy0xy1
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype="float32")
