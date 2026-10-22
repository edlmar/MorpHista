"""The three analyses. Each runs in a worker thread, writes its files next to the image and returns a Result."""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from .config import CELL_TYPES, IMAGE_EXTS, Config
from .detection import class_counts, detect_tiled
from .imaging import blend, hex_to_rgb, read_image, save_jpg, tissue_mask

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]

WHITE_ALIASES, GREY_ALIASES = ("white", "alba"), ("grey", "gray", "grisea")
CSV_FIELDS = [
    "Image",
    "Total_Area_mm2",
    "Tissue_Area_mm2",
    "Total_Area_px",
    "Tissue_Area_px",
    "Cells",
    "White_Matter",
    "Grey_Matter",
    "Classes",
    "Density_cells_per_mm2",
    "Pixel_Size_um",
    "Time_s",
]


@dataclass
class Result:
    title: str  # what was done, under the file name
    big: str  # headline figure
    unit: str  # caption under it
    summary: str  # one log line
    image: np.ndarray  # what the viewer shows
    rows: list[tuple[str, str]] = field(default_factory=list)  # extra label/value pairs
    table_row: dict | None = None  # Detect and count only


def collect_images(path_text: str) -> list[Path]:
    p = Path(path_text.strip().strip('"'))
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    return []


def _pixel_size(cfg: Config, from_file: float | None) -> float:
    return cfg.pixel_size_um if cfg.force_pixel_size or not from_file else from_file


def _out_dir(image: Path, name: str) -> Path:
    d = image.parent / name
    d.mkdir(exist_ok=True)
    return d


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def run_detection(path: Path, model, cfg: Config, progress: Progress, cancelled: Cancelled) -> Result:
    t0 = time.time()
    rgb, px_file = read_image(path)
    px = _pixel_size(cfg, px_file)
    progress(0.1, "tissue mask")
    mask = tissue_mask(rgb, px, cfg.threshold, cfg.min_area_mm2)
    tissue_px, total_px = cv2.countNonZero(mask), rgb.shape[0] * rgb.shape[1]

    progress(0.2, f"{model.kind} detection")
    det = detect_tiled(model, rgb, cfg, progress, cancelled, mask if cfg.skip_background else None)
    del mask
    counts = class_counts(det.class_id, model.names)
    white = sum(n for c, n in counts.items() if any(a in c.lower() for a in WHITE_ALIASES))
    grey = sum(n for c, n in counts.items() if any(a in c.lower() for a in GREY_ALIASES))

    progress(0.95, "saving")
    r, g, b = hex_to_rgb(cfg.box_color)  # supervision draws in BGR order, our image is RGB
    annotated = sv.BoxAnnotator(color=sv.Color(r=b, g=g, b=r), thickness=int(cfg.box_thickness)).annotate(
        rgb.copy(), det
    )
    out_dir = _out_dir(path, "Detections")
    save_jpg(out_dir / f"{path.stem}_Annotated.jpg", annotated)

    tissue_mm2 = tissue_px * px**2 / 1e6
    density = len(det) / tissue_mm2 if tissue_mm2 else 0.0
    row = {
        "Image": path.name,
        "Total_Area_mm2": round(total_px * px**2 / 1e6, 4),
        "Tissue_Area_mm2": round(tissue_mm2, 4),
        "Total_Area_px": total_px,
        "Tissue_Area_px": tissue_px,
        "Cells": len(det),
        "White_Matter": white,
        "Grey_Matter": grey,
        "Classes": "; ".join(f"{c}: {n}" for c, n in sorted(counts.items())),
        "Density_cells_per_mm2": round(density, 3),
        "Pixel_Size_um": round(px, 4),
        "Time_s": round(time.time() - t0, 2),
    }
    _append_csv(out_dir / "results.csv", row)

    rows = [("per mm² tissue", f"{density:.1f}"), ("tissue, mm²", f"{tissue_mm2:.4g}")]
    if white or grey:  # only meaningful for models with alba/grisea classes
        rows += [("white matter", str(white)), ("grey matter", str(grey))]
    rows.append(("µm per pixel", f"{px:.4g}"))
    return Result(
        title=CELL_TYPES.get(cfg.cell_type, cfg.cell_type),
        big=f"{len(det):,}".replace(",", " "),
        unit="cells detected",
        summary=f"{len(det)} cells, {density:.1f} per mm²",
        image=annotated,
        rows=rows,
        table_row=row,
    )


def run_tissue(path: Path, cfg: Config, progress: Progress, cancelled: Cancelled) -> Result:
    rgb, px_file = read_image(path)
    progress(0.3, "tissue mask")
    mask = tissue_mask(rgb, _pixel_size(cfg, px_file), cfg.threshold, cfg.min_area_mm2)
    out = blend(rgb, mask, np.array(hex_to_rgb(cfg.mask_color), np.float32), cfg.mask_alpha)
    save_jpg(_out_dir(path, "Masks") / f"{path.stem}_Mask.jpg", out, quality=80)
    share = cv2.countNonZero(mask) / mask.size
    return Result(
        title="Tissue mask",
        big=f"{share:.1%}",
        unit="of the image is tissue",
        summary=f"tissue {share:.1%} of image",
        image=out,
    )


def run_cell_seg(path: Path, model, cfg: Config, progress: Progress, cancelled: Cancelled) -> Result:
    rgb, _ = read_image(path)
    progress(0.3, f"{model.kind} segmentation")
    # Whole image in one pass (the model resizes internally); tile it like detect_tiled if small cells get lost
    det = model.predict(rgb, cfg.confidence)
    if det.mask is None:
        raise RuntimeError("model returned no masks — is this a segmentation checkpoint?")
    labels = np.zeros(rgb.shape[:2], np.uint16)
    for i, m in enumerate(det.mask[:65535], 1):
        labels[m] = i
    lut = (np.arange(len(det) + 1, dtype=np.uint32)[:, None] * [53, 97, 151] % 256).astype(np.uint8)
    lut[0] = 0
    overlay = blend(rgb, labels > 0, lut[labels], 0.45)

    progress(0.9, "saving")
    out_dir = _out_dir(path, "CellInstanceMasks")
    Image.fromarray(labels).save(out_dir / f"{path.stem}_CellInstances.png")
    Image.fromarray(np.where(labels > 0, 255, 0).astype(np.uint8)).save(out_dir / f"{path.stem}_CellMask.png")
    save_jpg(out_dir / f"{path.stem}_CellInstances_Overlay.jpg", overlay)
    n = int(labels.max())
    return Result(title="Cell segmentation", big=str(n), unit="cells outlined", summary=f"{n} instances", image=overlay)
