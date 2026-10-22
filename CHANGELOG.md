# Changelog

## 2.0.0 — 2026-10-22

Rewritten from scratch as a desktop application.

### Added
- Native desktop window built on NiceGUI, in light and dark themes.
- RF-DETR detection over overlapping tiles, with box merging, per-class counts and cell density per mm² of tissue.
- Support for RF-DETR `.pth` checkpoints and ONNX exports. Models are discovered automatically under `models/`.
- Tissue mask overlay, and RF-DETR-Seg cell instance masks.
- Zoomable result viewer (OpenSeadragon). It works at full resolution, has a minimap and a 1:1 zoom.
- Settings profiles, CSV export, and pixel size read from TIFF resolution tags.
- pixi environment with CUDA 12.8 (RTX 50xx), plus tasks for training and packaging.
- Release archive with the trained ICC model and `Start MorpHista.cmd`, which installs everything on the first start and puts a shortcut on the desktop.

### Performance
Measured on a 1.5-gigapixel slide with an RTX 5090:
- TIFF decoding is multithreaded: 17 s → 2 s.
- The mask overlay uses an OpenCV LUT: 72 s → 4 s.
- Detection is batched on the GPU and skips background tiles: 167 s → 59 s.
