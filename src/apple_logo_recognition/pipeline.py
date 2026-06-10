from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .detector import Detection, detect_apple_logos
from .io import read_rgb, write_rgb
from .visualization import draw_detections, mask_to_rgb


@dataclass(frozen=True)
class ProcessingResult:
    image: str
    detections: list[Detection]
    annotated_path: str


def run_image(image_path: str | Path, output_dir: str | Path = "results", min_score: float = 0.56) -> ProcessingResult:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image = read_rgb(image_path)
    detections, masks = detect_apple_logos(image, min_score=min_score)

    annotated = draw_detections(image, detections)
    annotated_path = output_dir / "annotated" / f"{image_path.stem}_detected.jpg"
    write_rgb(annotated_path, annotated)

    for name, mask in masks.items():
        write_rgb(output_dir / "masks" / f"{image_path.stem}_{name}.png", mask_to_rgb(mask))

    return ProcessingResult(str(image_path), detections, str(annotated_path))


def run_dataset(data_dir: str | Path = "data", output_dir: str | Path = "results", min_score: float = 0.56) -> list[ProcessingResult]:
    data_dir = Path(data_dir)
    image_paths = sorted(
        path for path in data_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_dir}")

    results = [run_image(path, output_dir=output_dir, min_score=min_score) for path in image_paths]
    _write_reports(results, Path(output_dir))
    return results


def _write_reports(results: list[ProcessingResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        for idx, detection in enumerate(result.detections, start=1):
            rows.append(
                {
                    "image": Path(result.image).name,
                    "detection_id": idx,
                    "score": round(detection.score, 4),
                    "polarity": detection.polarity,
                    "bbox_x1": detection.bbox[0],
                    "bbox_y1": detection.bbox[1],
                    "bbox_x2": detection.bbox[2],
                    "bbox_y2": detection.bbox[3],
                    "area": detection.features.area,
                    "aspect_ratio": round(detection.features.aspect_ratio, 4),
                    "fill_ratio": round(detection.features.fill_ratio, 4),
                    "perimeter_ratio": round(detection.features.perimeter_ratio, 4),
                }
            )

    fieldnames = [
        "image",
        "detection_id",
        "score",
        "polarity",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "area",
        "aspect_ratio",
        "fill_ratio",
        "perimeter_ratio",
    ]
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = [
        {
            "image": Path(result.image).name,
            "annotated_path": result.annotated_path,
            "detections": [
                {
                    "bbox": detection.bbox,
                    "score": detection.score,
                    "polarity": detection.polarity,
                    "features": asdict(detection.features),
                }
                for detection in result.detections
            ],
        }
        for result in results
    ]
    with (output_dir / "detections.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
