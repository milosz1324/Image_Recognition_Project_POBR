from __future__ import annotations

import numpy as np


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    image = image_rgb.astype(np.float32) / 255.0
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def percentile_normalize(gray: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(gray, [low, high])
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.float32)
    normalized = (gray - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def box_filter(image: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return image.astype(np.float32)
    padded = np.pad(image.astype(np.float32), radius, mode="reflect")
    result = np.zeros_like(image, dtype=np.float32)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return result / float(size * size)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    normalized = percentile_normalize(gray)
    local_mean = box_filter(normalized, radius=5)
    sharpened = normalized + 0.85 * (normalized - local_mean)
    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


def otsu_threshold(values: np.ndarray) -> float:
    clipped = np.clip(values, 0.0, 1.0)
    hist, _ = np.histogram(clipped.ravel(), bins=256, range=(0.0, 1.0))
    total = clipped.size
    if total == 0:
        return 0.5
    probabilities = hist.astype(np.float64) / total
    centers = (np.arange(256, dtype=np.float64) + 0.5) / 256.0
    cumulative_prob = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_prob * (1.0 - cumulative_prob)
    valid = denominator > 1e-12
    between = np.zeros_like(denominator)
    between[valid] = (global_mean * cumulative_prob[valid] - cumulative_mean[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def segment_bright_and_dark(enhanced_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    threshold = otsu_threshold(enhanced_gray)
    bright_level = max(threshold + 0.13, 0.72)
    dark_level = min(threshold - 0.13, 0.28)
    bright = enhanced_gray >= bright_level
    dark = enhanced_gray <= dark_level
    return bright, dark
