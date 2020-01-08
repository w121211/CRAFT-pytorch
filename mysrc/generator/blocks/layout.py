import random
from typing import List, Tuple, Union
import numpy as np

HOR = "HOR"
VER = "VER"
RAND = "RAND"


class Node:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.children = []

    def insert(self, node: "Node"):
        self.children.append(node)

    def leafs(self):
        leafs = []

        def _leafs(node: "Node"):
            if len(node.children) == 0:
                leafs.append(node)
            for c in node.children:
                _leafs(c)
        _leafs(self)

        return leafs


class Box(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rxy = np.zeros(2, dtype=np.int32)  # related xy
        self.xy: np.ndarray = None
        self.wh: np.ndarray = None

    def __repr__(self) -> str:
        c = self.children if len(self.children) > 0 else "NoChildren"
        return "<Box:{} {}>".format(self.bbox(), c)

    def set_xy(self, parent: Union["Box", None] = None):
        if parent is not None:
            self.xy = parent.xy + self.rxy
        elif self.xy is None:
            raise Exception("xy cannot be set")
        for c in self.children:
            c.set_xy(self)

    def bbox(self):
        if self.xy is None:
            return np.concatenate([self.rxy, self.rxy + self.wh])
        else:
            return np.concatenate([self.xy, self.xy + self.wh])

    def clone(self, cls, *args, **kwargs):
        root = cls(*args, **kwargs)
        root.rxy = self.rxy
        root.xy = self.xy
        root.wh = self.wh

        for c in self.children:
            c = c.clone(cls, *args, **kwargs)
            root.insert(c)
        return root


def align(boxes: List[Box], by: str, anchor: Union[float, None], dapart: int = 0) -> Tuple[int, int]:
    """
    Args:
        anchor: a float [0. to 1.], common settings such as 0.5 (center), 0. (top), 1. (bottom).
            If not provided, it will assume to be random.
    Return:
        a tuple of (w, h)
    """
    assert by in (HOR, VER)
    if len(boxes) == 0:
        return 0, 0

    if by == VER:
        if anchor is None:
            dxs = [np.random.normal(-0.5, 0.2) * b.wh[0] for b in boxes]
        else:
            dxs = [anchor * -b.wh[0] for b in boxes]

        dy = 0
        max_dx = -sorted(dxs)[0]
        for b, dx in zip(boxes, dxs):
            b.rxy = np.array([dx + max_dx, dy], dtype=np.int32)
            dy += b.wh[1] + dapart

        max_w = sorted([b.rxy[0] + b.wh[0] for b in boxes])[-1]
        return max_w, dy - dapart

    elif by == HOR:
        dys = (
            [np.random.normal(-0.5, 0.2) * b.wh[1] for b in boxes]
            if anchor is None
            else [anchor * -b.wh[1] for b in boxes]
        )
        dx = 0
        max_dy = -sorted(dys)[0]
        for b, dy in zip(boxes, dys):
            b.rxy = np.array([dx, dy + max_dy], dtype=np.int32)
            dx += b.wh[0] + dapart
        max_h = sorted([b.rxy[1] + b.wh[1] for b in boxes])[-1]
        return dx - dapart, max_h


class BoxTemplate(Box):
    def __init__(self, align=None, children=[], ratio=[]):
        super().__init__()
        for c in children:
            if callable(c):
                c = c()
            if not isinstance(c, BoxTemplate):
                raise Exception
            self.insert(c)
        self.align = align
        self.ratio = ratio

        r = [0] + ratio + [1]
        rr = np.repeat(np.expand_dims(r, axis=1), 2, axis=1)
        mask_x = np.array([1, 0])
        mask_y = np.array([0, 1])
        mask1, mask2 = (mask_x, mask_y) if align == HOR else (mask_y, mask_x)
        _rxy0 = rr[:-1] * mask1
        _rxy1 = rr[1:] * mask1 + np.array([1, 1]) * mask2
        _wh = _rxy1 - _rxy0

        self._rxy0 = _rxy0
        self._wh = _wh

        # print("-------")
        # print(r)
        # print(rr)
        # print(self._rxy0)
        # print(_rxy1)
        # print(self._wh)

    def one(self, wh: np.ndarray):
        root = Box()
        root.wh = wh
        for c, _rxy0, _wh in zip(self.children, self._rxy0, self._wh):
            b = c.one((_wh * wh).astype(np.int32))
            b.rxy = (_rxy0 * wh).astype(np.int32)
            root.insert(b)
        return root


# ratios
r2 = [[1 / 2], [1 / 3], [2 / 3], [1 / 4], [3 / 4]]
r3 = [[1 / 3, 2 / 3], [1 / 4, 2 / 4],
      [1 / 4, 3 / 4], [2 / 4, 3 / 4], [1 / 3, 3 / 4]]

# 2-box
b2s = lambda x2, x3: [
    [b, x2], [x2, b], [b, x3], [x3, b],
    [x2, x2], [x3, x3], [x2, x3], [x3, x2]
]

# 3-box
b3s = lambda x2, x3: [
    [b, b, x2], [b, x2, b], [x2, b, b],
    [b, b, x3], [b, x3, b], [x3, b, b],
    [b, x2, x3], [x2, b, x3], [x2, x3, b],
    [b, x3, x2], [x3, b, x2], [x3, x2, b],
    [x2, x3, x2], [x3, x2, x3]
]

b = lambda: BoxTemplate()
bh2 = lambda: BoxTemplate(HOR, [b, b], random.choice(r2))
bh3 = lambda: BoxTemplate(HOR, [b, b, b], random.choice(r3))
# bvh2 = lambda: BoxTemplate(VER, random.choice(
#     b2s(bh2, bh3)), random.choice(r2))
bvh2 = lambda: BoxTemplate(VER, random.choice(
    b2s(b, bh2)), random.choice(r2))
bvh3 = lambda: BoxTemplate(VER, random.choice(
    b3s(bh2, bh3)), random.choice(r3))


def rand_box(wh: np.ndarray,
             choices=[bvh2]
             # choices=[bh2, bh3, bvh2, bvh3]
             ):
    bt = random.choice(choices)()
    # bt = random.choice([bvh3])()
    # bt = BoxTemplate(VER, [BoxTemplate(), BoxTemplate(
    #     HOR, [BoxTemplate(), BoxTemplate()], [1 / 2]
    # )], [1 / 2])
    return bt.one(wh)
