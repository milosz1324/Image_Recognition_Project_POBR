from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .components import Component, connected_components
from .features import ShapeFeatures, apple_bite_score, apple_body_template, compute_shape_features, template_iou
from .morphology import closing, fill_holes, opening
from .preprocessing import enhance_contrast, rgb_to_gray, segment_bright_and_dark


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    polarity: str
    features: ShapeFeatures


def prepare_masks(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    gray = rgb_to_gray(image_rgb)
    enhanced = enhance_contrast(gray)
    bright, dark = segment_bright_and_dark(enhanced)
    rgb = image_rgb.astype(np.float32) / 255.0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    warm_bright = (gray > 0.45) & ((red + green) / 2.0 - blue > 0.07) & (red > 0.45) & (green > 0.40)
    masks = {
        "bright": fill_holes(closing(opening(bright, radius=1), radius=2)),
        "dark": fill_holes(closing(opening(dark, radius=1), radius=2)),
        "warm_bright": fill_holes(closing(opening(warm_bright, radius=1), radius=2)),
    }
    return masks


def _candidate_score(component: Component, image_shape: tuple[int, int], template: np.ndarray) -> tuple[float, ShapeFeatures]:
    h, w = image_shape
    features = compute_shape_features(component.mask, component.bbox)
    area_fraction = component.area / float(h * w)
    if area_fraction < 0.0015 or area_fraction > 0.18:
        return 0.0, features
    if component.width < 42 or component.height < 42:
        return 0.0, features
    if not (0.45 <= features.aspect_ratio <= 1.65):
        return 0.0, features
    if not (0.22 <= features.fill_ratio <= 0.92):
        return 0.0, features
    if features.perimeter_ratio > 10.0 and features.fill_ratio < 0.42:
        return 0.0, features

    iou = template_iou(component.mask, template)
    bite_score = apple_bite_score(component.mask)
    aspect_score = max(0.0, 1.0 - abs(features.aspect_ratio - 0.95) / 0.75)
    fill_score = max(0.0, 1.0 - abs(features.fill_ratio - 0.58) / 0.45)
    perimeter_score = max(0.0, 1.0 - abs(features.perimeter_ratio - 5.4) / 6.5)
    score = 0.44 * iou + 0.28 * bite_score + 0.12 * aspect_score + 0.09 * fill_score + 0.07 * perimeter_score
    return float(score), features


def _non_max_suppression(detections: list[Detection], threshold: float = 0.25) -> list[Detection]:
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    selected: list[Detection] = []
    for detection in ordered:
        if all(_bbox_iou(detection.bbox, other.bbox) < threshold for other in selected):
            selected.append(detection)
    return selected


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / float(area_a + area_b - inter)


def detect_apple_logos(image_rgb: np.ndarray, min_score: float = 0.56) -> tuple[list[Detection], dict[str, np.ndarray]]:
    masks = prepare_masks(image_rgb)
    template = apple_body_template()
    detections: list[Detection] = []
    image_shape = image_rgb.shape[:2]

    for polarity, mask in masks.items():
        min_area = max(25, int(mask.size * 0.00015))
        for component in connected_components(mask, min_area=min_area):
            score, features = _candidate_score(component, image_shape, template)
            if score >= min_score:
                detections.append(Detection(component.bbox, score, polarity, features))

    return _non_max_suppression(detections), masks
