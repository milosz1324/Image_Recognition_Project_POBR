from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ShapeFeatures:
    area: int
    bbox: tuple[int, int, int, int]
    aspect_ratio: float
    fill_ratio: float
    perimeter_ratio: float
    hu_log: tuple[float, ...]

    def vector(self) -> np.ndarray:
        return np.array(
            [self.aspect_ratio, self.fill_ratio, self.perimeter_ratio, *self.hu_log[:4]],
            dtype=np.float32,
        )


def resize_binary_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros((size, size), dtype=bool)
    y_idx = np.minimum((np.arange(size) * mask.shape[0] / size).astype(int), mask.shape[0] - 1)
    x_idx = np.minimum((np.arange(size) * mask.shape[1] / size).astype(int), mask.shape[1] - 1)
    return mask[y_idx[:, None], x_idx[None, :]].astype(bool)


def square_pad(mask: np.ndarray, margin: float = 0.12) -> np.ndarray:
    h, w = mask.shape
    side = int(max(h, w) * (1.0 + 2.0 * margin))
    out = np.zeros((side, side), dtype=bool)
    y = (side - h) // 2
    x = (side - w) // 2
    out[y : y + h, x : x + w] = mask
    return out


def perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return int(np.count_nonzero(center & ~interior))


def compute_shape_features(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> ShapeFeatures:
    mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    h, w = mask.shape
    if area == 0 or h == 0 or w == 0:
        return ShapeFeatures(0, bbox, 0.0, 0.0, 0.0, (0.0,) * 7)

    ys, xs = np.nonzero(mask)
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    x = xs.astype(np.float64) - x_mean
    y = ys.astype(np.float64) - y_mean
    m00 = float(area)

    def mu(p: int, q: int) -> float:
        return float(np.sum((x**p) * (y**q)))

    def eta(p: int, q: int) -> float:
        return mu(p, q) / (m00 ** (1.0 + (p + q) / 2.0))

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    hu = (
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12)
        * (n30 + n12)
        * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03)
        * (n21 + n03)
        * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
        + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03)
        * (n30 + n12)
        * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12)
        * (n21 + n03)
        * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    )
    hu_log = tuple(float(-np.sign(v) * np.log10(abs(v) + 1e-12)) for v in hu)
    aspect = w / max(h, 1)
    fill = area / float(w * h)
    perimeter_ratio = perimeter(mask) / max(np.sqrt(area), 1.0)
    return ShapeFeatures(area, bbox, aspect, fill, perimeter_ratio, hu_log)


def apple_body_template(size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    x = (xx + 0.5) / size
    y = (yy + 0.5) / size
    left_lobe = ((x - 0.40) / 0.25) ** 2 + ((y - 0.53) / 0.34) ** 2 <= 1.0
    right_lobe = ((x - 0.61) / 0.25) ** 2 + ((y - 0.53) / 0.34) ** 2 <= 1.0
    lower = ((x - 0.50) / 0.34) ** 2 + ((y - 0.68) / 0.31) ** 2 <= 1.0
    top = ((x - 0.50) / 0.27) ** 2 + ((y - 0.39) / 0.22) ** 2 <= 1.0
    bite = ((x - 0.73) / 0.15) ** 2 + ((y - 0.42) / 0.18) ** 2 <= 1.0
    notch = ((x - 0.52) / 0.11) ** 2 + ((y - 0.25) / 0.10) ** 2 <= 1.0
    return (left_lobe | right_lobe | lower | top) & ~bite & ~notch


def template_iou(mask: np.ndarray, template: np.ndarray) -> float:
    candidate = resize_binary_nearest(square_pad(mask), template.shape[0])
    union = np.count_nonzero(candidate | template)
    if union == 0:
        return 0.0
    direct = np.count_nonzero(candidate & template) / union
    flipped = np.count_nonzero(candidate[:, ::-1] & template) / np.count_nonzero(candidate[:, ::-1] | template)
    return float(max(direct, flipped))


def apple_bite_score(mask: np.ndarray, size: int = 96) -> float:
    candidate = resize_binary_nearest(square_pad(mask), size)
    right_edges = np.full(size, -1, dtype=np.int32)
    for y in range(size):
        xs = np.flatnonzero(candidate[y])
        if xs.size:
            right_edges[y] = int(xs.max())

    def median_edge(start: int, stop: int) -> float | None:
        values = right_edges[start:stop]
        values = values[values >= 0]
        if values.size == 0:
            return None
        return float(np.median(values))

    upper = median_edge(int(size * 0.25), int(size * 0.36))
    bite = median_edge(int(size * 0.38), int(size * 0.55))
    lower = median_edge(int(size * 0.58), int(size * 0.78))
    if upper is None or bite is None or lower is None:
        return 0.0

    outer_edge = max(upper, lower)
    indentation = max(0.0, outer_edge - bite) / size
    return float(np.clip(indentation / 0.12, 0.0, 1.0))
