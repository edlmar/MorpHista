"""Tiled detection over large images and the post-processing of the boxes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import cv2
import numpy as np
import supervision as sv

from .config import Config

Tile = tuple[int, int, int, int]  # x1, y1, x2, y2


def make_tiles(h: int, w: int, size: int, overlap: int) -> list[Tile]:
    """Equal windows that cover the image exactly; the last row/column is shifted back to the edge."""

    def starts(n: int) -> tuple[list[int], int]:
        win = min(size, n)
        s = list(range(0, n - win + 1, max(1, win - overlap)))
        if s[-1] != n - win:
            s.append(n - win)
        return s, win

    xs, ww = starts(w)
    ys, wh = starts(h)
    return [(x, y, x + ww, y + wh) for y in ys for x in xs]


def tissue_tiles(tiles: list[Tile], mask: np.ndarray | None, min_frac: float = 0.01) -> list[Tile]:
    """Drop tiles with (almost) no tissue. Keeps everything if the mask found next to no tissue,
    e.g. a dark-background image the brightness cut-off doesn't suit."""
    if mask is None or cv2.countNonZero(mask) < min_frac * mask.size:
        return tiles
    return [
        t for t in tiles if cv2.countNonZero(mask[t[1] : t[3], t[0] : t[2]]) >= min_frac * (t[2] - t[0]) * (t[3] - t[1])
    ]


def suppress_close(det: sv.Detections, px: int) -> sv.Detections:
    """Drop lower-confidence boxes of the same class that touch or sit within px of a kept box."""
    if px <= 0 or len(det) < 2:
        return det
    b = det.xyxy
    conf = det.confidence if det.confidence is not None else np.ones(len(det))
    cls = det.class_id if det.class_id is not None else np.full(len(det), -1)
    centers = (b[:, :2] + b[:, 2:]) / 2
    alive = np.ones(len(det), bool)
    keep = []
    for i in np.argsort(-conf):  # O(n²) but vectorised per row: fine up to ~50k boxes
        if not alive[i]:
            continue
        keep.append(i)
        gx = np.maximum(0, np.maximum(b[:, 0] - b[i, 2], b[i, 0] - b[:, 2]))
        gy = np.maximum(0, np.maximum(b[:, 1] - b[i, 3], b[i, 1] - b[:, 3]))
        close = ((gx <= px) & (gy <= px)) | (np.hypot(*(centers - centers[i]).T) <= 1.5 * px)
        same = (cls == cls[i]) | (cls < 0) | (cls[i] < 0)
        alive &= ~(close & same)
    return det[np.array(keep)]


def class_counts(class_ids, names: dict[int, str]) -> Counter:
    ids = [int(c) for c in (class_ids if class_ids is not None else [])]
    # Roboflow COCO exports shift ids by one vs. class_names; pick the offset that matches best.
    off = max((0, -1, 1), key=lambda o: sum((c + o) in names for c in ids)) if names else 0
    return Counter(names.get(c + off) or names.get(c) or f"class_{c}" for c in ids)


def detect_tiled(
    model,
    rgb: np.ndarray,
    cfg: Config,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    mask: np.ndarray | None = None,
) -> sv.Detections:
    """Run the model over overlapping tiles (only tissue tiles if `mask` is given), then merge."""
    h, w = rgb.shape[:2]
    size = cfg.tile_size or model.resolution
    overlap = cfg.tile_overlap or int(np.clip(size * 0.15, 32, 256))
    all_tiles = make_tiles(h, w, size, min(overlap, size - 1))
    tiles = tissue_tiles(all_tiles, mask)
    step, n, parts = model.BATCH, len(tiles), []
    for i in range(0, n, step):
        if cancelled():
            return sv.Detections.empty()
        chunk = tiles[i : i + step]
        found = model.predict_batch([rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in chunk], cfg.confidence)
        for (x1, y1, _, _), d in zip(chunk, found, strict=True):
            if len(d):
                parts.append(
                    sv.Detections(
                        xyxy=d.xyxy + np.array([x1, y1, x1, y1], np.float32),
                        confidence=d.confidence,
                        class_id=d.class_id,
                    )
                )
        done = min(i + step, n)
        progress(0.2 + 0.7 * done / max(n, 1), f"tile {done}/{n} with tissue (of {len(all_tiles)}) @ {size}px")
    det = sv.Detections.merge(parts) if parts else sv.Detections.empty()
    if len(det) > 1:
        det = det.with_nmm(threshold=cfg.nms, class_agnostic=False)
    return suppress_close(det, cfg.merge_px)
