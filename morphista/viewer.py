"""Deep Zoom tiles of the latest result for the OpenSeadragon viewer: full resolution at any zoom."""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
from fastapi import Response
from nicegui import app

from .imaging import encode_jpg

TILE = 512


class Pyramid:
    """Levels are built on demand by halving the one above.

    Limit: keeps the full-res RGB (+1/3 for lower levels) in RAM, ~6 GB for a 1.5 Gpx slide;
    write the pyramid to disk (pyvips dzsave) if that ever gets too heavy.
    """

    def __init__(self, rgb: np.ndarray):
        self.h, self.w = rgb.shape[:2]
        self.max_level = math.ceil(math.log2(max(self.w, self.h, 2)))
        self.levels = {self.max_level: rgb}
        self.key = str(time.time_ns())  # new URL per result, so the browser cache never mixes images
        self.lock = threading.RLock()

    def level(self, n: int) -> np.ndarray:
        with self.lock:
            if n not in self.levels:
                up = self.level(n + 1)
                size = ((up.shape[1] + 1) // 2, (up.shape[0] + 1) // 2)  # DZI: each level is ceil(half)
                self.levels[n] = cv2.resize(up, size, interpolation=cv2.INTER_AREA)
            return self.levels[n]

    def tile(self, n: int, col: int, row: int) -> bytes | None:
        y, x = row * TILE, col * TILE
        part = self.level(n)[y : y + TILE, x : x + TILE]
        return encode_jpg(part, 90) if part.size else None


_current: Pyramid | None = None


def publish(rgb: np.ndarray) -> Pyramid:
    """Make `rgb` the image the viewer serves; the previous one is released."""
    global _current
    _current = Pyramid(rgb)
    return _current


@app.get("/tiles/{key}/{level}/{tile}.jpg")
def serve_tile(key: str, level: int, tile: str) -> Response:
    p = _current
    col, _, row = tile.partition("_")
    data = p.tile(level, int(col), int(row)) if p and p.key == key and 0 <= level <= p.max_level else None
    if data is None:
        return Response(status_code=404)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})
