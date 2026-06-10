from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def connected_components(mask: np.ndarray, min_area: int = 20) -> list[Component]:
    mask = mask.astype(bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    components: list[Component] = []
    label = 0

    for start_y in range(h):
        for start_x in range(w):
            if not mask[start_y, start_x] or labels[start_y, start_x] != 0:
                continue
            label += 1
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            labels[start_y, start_x] = label
            pixels: list[tuple[int, int]] = []

            while queue:
                y, x = queue.popleft()
                pixels.append((y, x))
                for ny in (y - 1, y, y + 1):
                    for nx in (x - 1, x, x + 1):
                        if ny == y and nx == x:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            queue.append((ny, nx))

            if len(pixels) < min_area:
                continue
            ys = np.array([p[0] for p in pixels], dtype=np.int32)
            xs = np.array([p[1] for p in pixels], dtype=np.int32)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            component_mask = labels[y1 : y2 + 1, x1 : x2 + 1] == label
            components.append(Component(label, len(pixels), (x1, y1, x2, y2), component_mask))

    return components
