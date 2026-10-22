"""Image I/O, tissue mask and colour overlays. Tuned for whole-slide sizes (~1.5 Gpx)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logging.getLogger("tifffile").setLevel(logging.ERROR)  # noisy about odd OME metadata


def _tiff_pixel_size_um(im: Image.Image) -> float | None:
    tags = getattr(im, "tag_v2", None)
    if not tags or 282 not in tags:
        return None
    xres = float(tags[282])
    per_unit_um = {2: 25400.0, 3: 10000.0}.get(tags.get(296, 1))  # inch / cm; no unit -> unknown
    return per_unit_um / xres if per_unit_um and xres > 0 else None


def read_image(path: Path) -> tuple[np.ndarray, float | None]:
    """RGB uint8 array + pixel size in µm from TIFF resolution tags (if any)."""
    with Image.open(path) as im:  # PIL opens lazily: tags for every format, pixels for non-TIFF
        px = _tiff_pixel_size_um(im)
        if path.suffix.lower() not in (".tif", ".tiff"):
            return np.array(im.convert("RGB")), px
    try:  # tifffile decodes TIFF tiles/strips on all cores: ~10x PIL on a 1.5 Gpx slide
        arr = tifffile.imread(path, key=0, maxworkers=os.cpu_count())
    except Exception:
        with Image.open(path) as im:
            return np.array(im.convert("RGB")), px
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)  # planar (C, H, W) storage
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(arr[..., :3]), px


def encode_jpg(rgb: np.ndarray, quality: int) -> bytes:
    """libjpeg-turbo via OpenCV: ~3x PIL."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 85) -> None:
    if max(rgb.shape[:2]) > 65500:  # JPEG hard limit
        f = 65500 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    path.write_bytes(encode_jpg(rgb, quality))  # write_bytes copes with non-ASCII paths, cv2.imwrite doesn't


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def tissue_mask(rgb: np.ndarray, pixel_um: float, threshold: int, min_area_mm2: float) -> np.ndarray:
    """255 where the slide is darker than `threshold`, closed, minus pieces smaller than `min_area_mm2`."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    k = int(np.clip(pixel_um * 10, 5, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if min_area_mm2 > 0 and pixel_um > 0:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area_mm2 * 1e6 / pixel_um**2
        keep[0] = False
        mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    return mask


def blend(rgb: np.ndarray, where: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    """Tint pixels where `where` is non-zero with one RGB colour or a per-pixel colour image.

    OpenCV LUT + masked copy instead of numpy boolean indexing: 72 s -> 4 s on a 1.5 Gpx slide.
    """
    if np.ndim(color) == 1:
        lut = np.arange(256, dtype=np.float32)[:, None] * (1 - alpha) + np.asarray(color, np.float32) * alpha
        tinted = cv2.LUT(rgb, np.clip(np.round(lut), 0, 255).astype(np.uint8).reshape(256, 1, 3))
    else:
        tinted = cv2.addWeighted(rgb, 1 - alpha, color, alpha, 0)
    out = rgb.copy()
    cv2.copyTo(tinted, where.view(np.uint8) if where.dtype == bool else where, out)
    return out
