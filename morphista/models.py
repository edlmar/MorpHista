"""Detection models (RF-DETR checkpoints, ONNX exports) and their discovery under models/."""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

from .config import MODELS_DIR

CUDA = torch.cuda.is_available()
DEVICE_LABEL = f"CUDA: {torch.cuda.get_device_name(0)}" if CUDA else "CPU, no CUDA"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

CELL_KEYWORDS = {"icc": ("icc", "cajal", "intestinal"), "astrocytes": ("astro",)}
SEG_MARKERS = ("/seg/", "/segment", "-seg", "_seg")


def parse_class_names(raw) -> dict[int, str]:
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = [s.strip() for s in raw.split(",") if s.strip()]
    items = raw.items() if isinstance(raw, dict) else enumerate(raw or [])
    return {int(k): str(v) for k, v in items}


def decode_rfdetr(boxes: np.ndarray, logits: np.ndarray, w: int, h: int, thr: float, top_k: int = 300) -> sv.Detections:
    """RF-DETR head output (normalized cxcywh boxes, sigmoid logits) -> pixel detections."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    flat = prob.ravel()
    k = min(top_k, flat.size)
    top = np.argpartition(-flat, k - 1)[:k]
    top = top[flat[top] >= thr]
    query, cls = np.divmod(top, prob.shape[1])
    cx, cy, bw, bh = boxes[query].T
    xyxy = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    xyxy = np.clip(xyxy, 0, [w, h, w, h]).astype(np.float32).reshape(-1, 4)
    return sv.Detections(xyxy=xyxy, confidence=flat[top].astype(np.float32), class_id=cls.astype(int))


class OnnxDetector:
    """RF-DETR exported to ONNX; runs on CUDA, DirectML or CPU, whichever onnxruntime offers."""

    BATCH = 1

    def __init__(self, path: Path):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.session = ort.InferenceSession(str(path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name, self.fp16 = inp.name, "float16" in inp.type
        # RF-DETR exports a static input size; dynamic dims fall back to 640
        self.h, self.w = (d if isinstance(d, int) else 640 for d in inp.shape[2:4])
        self.out_names = [o.name.lower() for o in self.session.get_outputs()]
        self.resolution = max(self.h, self.w)
        self.names: dict[int, str] = {}
        self.kind = f"ONNX ({self.session.get_providers()[0]})"

    def predict(self, rgb: np.ndarray, thr: float) -> sv.Detections:
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        outs = self.session.run(None, {self.input_name: x.astype(np.float16 if self.fp16 else np.float32)})
        box_idx = next((i for i, n in enumerate(self.out_names) if "box" in n or "det" in n), 0)
        boxes = np.asarray(outs[box_idx], np.float32)[0]
        logits = np.asarray(outs[1 - box_idx], np.float32)[0]
        return decode_rfdetr(boxes, logits, w, h, thr)

    def predict_batch(self, tiles: list[np.ndarray], thr: float) -> list[sv.Detections]:
        return [self.predict(t, thr) for t in tiles]


class RFDETRModel:
    """A trained RF-DETR .pth; the architecture is rebuilt from the args stored in the checkpoint."""

    # measured on an RTX 5090: 4 tiles per call is fastest; bigger batches stall in rfdetr's CPU post-processing
    BATCH = 4
    SIZES = [
        ("2x", "2XLarge"),
        ("xxlarge", "2XLarge"),
        ("xlarge", "XLarge"),
        ("large", "Large"),
        ("base", "Base"),
        ("medium", "Medium"),
        ("small", "Small"),
        ("nano", "Nano"),
    ]
    ARG_KEYS = {
        "encoder",
        "out_feature_indexes",
        "dec_layers",
        "two_stage",
        "projector_scale",
        "hidden_dim",
        "patch_size",
        "num_windows",
        "sa_nheads",
        "ca_nheads",
        "dec_n_points",
        "bbox_reparam",
        "lite_refpoint_refine",
        "layer_norm",
        "amp",
        "group_detr",
        "gradient_checkpointing",
        "positional_encoding_size",
        "resolution",
        "num_queries",
        "num_select",
    }

    def __init__(self, path: Path, seg: bool):
        import rfdetr

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)  # local trusted checkpoints
        args = ckpt.get("args") or {}
        args = args if isinstance(args, dict) else vars(args)
        self.names = parse_class_names(args.get("class_names"))

        prefix = "RFDETRSeg" if seg else "RFDETR"
        hay = " ".join(
            [str(path)]
            + [str(args.get(k, "")) for k in ("pretrain_weights", "encoder", "model", "model_name", "output_dir")]
        ).lower()
        candidates = [s for marker, s in self.SIZES if marker in hay] + ["Large", "Base", "Medium"]
        model_cls = next((c for c in (getattr(rfdetr, prefix + s, None) for s in candidates) if c), None)
        if model_cls is None:
            raise RuntimeError(f"installed rfdetr has no {prefix}* model classes")

        keys = self.ARG_KEYS | ({"mask_downsample_ratio"} if seg else set())
        kwargs = {k: args[k] for k in keys if args.get(k) is not None}
        if seg:
            kwargs["segmentation_head"] = True
        if self.names:
            kwargs["num_classes"] = len(self.names)
        elif isinstance(args.get("num_classes"), int) and args["num_classes"] > 1:
            kwargs["num_classes"] = args["num_classes"] - 1  # checkpoint counts background
        self.model = model_cls(pretrain_weights=str(path), device="cuda" if CUDA else "cpu", **kwargs)

        optimize = getattr(self.model, "inference", None) or self.model.optimize_for_inference  # renamed in 1.11
        try:
            optimize(compile=False, batch_size=1, dtype=torch.float16 if CUDA else torch.float32)
            mode = "FP16" if CUDA else "FP32"
        except Exception as e:  # still usable, just slower
            mode = f"not optimized ({e})"
        model_config = getattr(self.model, "model_config", None)
        self.resolution = int(args.get("resolution") or getattr(model_config, "resolution", 560))
        self.kind = f"RF-DETR {model_cls.__name__}, {mode}"

    def predict(self, rgb: np.ndarray, thr: float) -> sv.Detections:
        """Whole image as numpy: rfdetr resizes it on the CPU (a full slide as a GPU float tensor is ~18 GB)."""
        with torch.inference_mode():
            det = self.model.predict(rgb, threshold=thr)
        return det[0] if isinstance(det, list) else det

    def predict_batch(self, tiles: list[np.ndarray], thr: float) -> list[sv.Detections]:
        if CUDA:  # ready GPU tensors skip rfdetr's per-image CPU conversion (+30 % throughput)
            tiles = [torch.from_numpy(np.ascontiguousarray(t)).cuda().permute(2, 0, 1).float().div_(255) for t in tiles]
        with torch.inference_mode():
            det = self.model.predict(tiles if len(tiles) > 1 else tiles[0], threshold=thr)
        return det if isinstance(det, list) else [det]


_loaded: dict[tuple[str, bool], OnnxDetector | RFDETRModel] = {}


def get_model(path: str, seg: bool) -> OnnxDetector | RFDETRModel:
    """Load once, keep only the most recently used model in memory."""
    key = (str(Path(path).resolve()), seg)
    if key not in _loaded:
        _loaded.clear()
        if CUDA:
            torch.cuda.empty_cache()
        p = Path(key[0])
        _loaded[key] = OnnxDetector(p) if p.suffix.lower() == ".onnx" else RFDETRModel(p, seg)
    return _loaded[key]


def list_models(cell_type: str, segmentation: bool) -> list[Path]:
    """.pth/.onnx under models/ for this cell type; names tell cell type (icc, astro) and task (seg)."""
    found = []
    if MODELS_DIR.is_dir():
        for p in MODELS_DIR.rglob("*"):
            suffix = p.suffix.lower()
            if suffix not in (".pth", ".onnx") or (segmentation and suffix == ".onnx"):
                continue
            s = "/" + p.relative_to(MODELS_DIR).as_posix().lower()  # ignore folders above models/
            if any(m in s for m in SEG_MARKERS) != segmentation:
                continue
            owner = next((ct for ct, kws in CELL_KEYWORDS.items() if any(k in s for k in kws)), None)
            if owner in (None, cell_type):
                found.append(p)
    return sorted(found, key=lambda p: str(p).lower())
