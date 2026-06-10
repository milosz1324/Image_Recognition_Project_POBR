from __future__ import annotations

import numpy as np

from .detector import Detection


def draw_detections(image_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_rgb.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = np.array([20, 220, 60], dtype=np.uint8) if detection.polarity == "bright" else np.array([255, 60, 40], dtype=np.uint8)
        thickness = max(2, min(out.shape[:2]) // 220)
        out[y1 : y1 + thickness, x1 : x2 + 1] = color
        out[y2 - thickness + 1 : y2 + 1, x1 : x2 + 1] = color
        out[y1 : y2 + 1, x1 : x1 + thickness] = color
        out[y1 : y2 + 1, x2 - thickness + 1 : x2 + 1] = color
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask.astype(bool)] = np.array([255, 255, 255], dtype=np.uint8)
    return image
