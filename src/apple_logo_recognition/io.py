from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image using OpenCV and return an RGB uint8 array."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image_bgr[:, :, ::-1].copy()


def write_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Write an RGB uint8 image using OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = np.asarray(image_rgb)[:, :, ::-1]
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise OSError(f"Cannot write image: {path}")
