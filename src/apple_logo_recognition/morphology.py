from __future__ import annotations

from collections import deque

import numpy as np


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def opening(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return dilate(erode(mask, radius), radius)


def closing(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    return erode(dilate(mask, radius), radius)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    background = ~mask
    visited = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    queue: deque[tuple[int, int]] = deque()

    for x in range(w):
        for y in (0, h - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for ny in (y - 1, y, y + 1):
            for nx in (x - 1, x, x + 1):
                if ny == y and nx == x:
                    continue
                if 0 <= ny < h and 0 <= nx < w and background[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    holes = background & ~visited
    return mask | holes
