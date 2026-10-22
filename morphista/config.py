"""Paths, analysis settings and the settings file."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path

# next to the .exe when packaged, the repository root otherwise
APP_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "models"
SETTINGS_FILE = Path.home() / ".morphista" / "settings.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".jp2", ".png", ".tif", ".tiff", ".webp"}
CELL_TYPES = {"icc": "Intestinal cells of Cajal", "astrocytes": "Astrocytes"}


@dataclass
class Config:
    """Everything that affects the analysis. Profiles save and load exactly this."""

    cell_type: str = "icc"
    model_path: str | None = None
    seg_model_path: str | None = None
    input_path: str = ""
    # tissue
    threshold: int = 210
    min_area_mm2: float = 0.0
    pixel_size_um: float = 0.44
    force_pixel_size: bool = False
    mask_color: str = "#10D98A"
    mask_alpha: float = 0.3
    # detection
    confidence: float = 0.1
    nms: float = 0.25
    merge_px: int = 12
    tile_size: int = 0  # 0 = model input resolution
    tile_overlap: int = 0  # 0 = 15 % of the tile, clamped to 32..256
    skip_background: bool = True  # run the model only on tiles with >= 1 % tissue
    box_color: str = "#10D98A"  # green separates from both hematoxylin blue and DAB brown
    box_thickness: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def load_settings() -> dict:
    """{"config": {...}, "profiles": {name: {...}}, "dark": bool | None}"""
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
