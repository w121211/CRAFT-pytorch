from typing import List, Tuple, Union
import numpy as np

HOR = "HOR"
VER = "VER"
RAND = "RAND"


class Node:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = None
        self.children = []

    def insert(self, node: "Node"):
        self.children.append(node)


class Box(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rxy = np.zeros(2, dtype=np.int32)  # related xy
        self.xy: np.ndarray = None
        self.wh: np.ndarray = None

    def set_xy(self, parent: Union["Box", None] = None):
        if parent is not None:
            self.xy = parent.xy + self.rxy
        elif self.xy is None:
            raise Exception("xy cannot be set")
        for c in self.children:
            c.set_xy(self)


class BoxTemplate(Box):
    def __init__(self, align=None, children=[], ratio=[]):
        self.align = align
        self.children = children
        self.ratio = ratio

        r = [0] + ratio + [1]
        rr = np.repeat(np.expand_dims(r, axis=1), 2, axis=1)
        mask = np.array([1, 0]) if align == HOR else np.array([0, 1])
        _rxy0 = rr[:-1] * mask
        _rxy1 = rr[1:] * mask + np.array([0, 1])
        _wh = _rxy1 - _rxy0

        self._rxy0 = _rxy0
        self._wh = _wh

        print(self._rxy0)
        print(self._wh)

    def spawn(self, wh: np.ndarray):
        root = Box()
        root.wh = wh

        for c, _rxy0, _wh in zip(self.children, self._rxy0, self._wh):
            b = c.spawn(_wh * wh)
            b.rxy = (_rxy0 * wh).astype(np.int32)
            root.insert(b)
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


def rand_boxes() -> List[Box]:
    pass

# r2 = [1/2, 1/3, 2/3, 1/4, 4/3]
# r3 = [[1/3, 2/3], [1/4, 2/4], [1/4, 3/4]]

# b = BT()
# bh2 = BT(HOR, [BT(), BT()], choice(r2))
# bh3 = BT(HOR, [BT(), BT(), BT()], choice(r2))

# _bh2 = [[b, bh2], [b, bh3], [bh2, b], [bh3, b], [bh2, bh2], [bh3, bh3], [_bh2, _bh3], , [bh3, bh2]]
# _bh3 = [[b, bh2, b], [b, bh3, b], [b, bh2, bh3], [], [_bh3, _b], [bh3, bh2]]

# bvh2 = BT(VER, [_bh2, _bh2])
# bhv2 = BT(HOR, [bv2, bv2], choice(r2))
# bhv3 = BT(HOR, [bv2, bv2], choice(r2))
