from __future__ import annotations

import os
from pathlib import Path

import yaml

os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "data.yaml"

MODEL_H = 312
MODEL_W = 312
INPUT_SHAPE = (1, 3, MODEL_H, MODEL_W)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

DEFAULT_ENGINE = PROJECT_ROOT / "exports" / "rfdetr" / "rfdetr-seg-nano.engine"
DEFAULT_INPUT = PROJECT_ROOT / "test"
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "runs" / "tensorrt_raw_outputs"
DEFAULT_MASK_OUTPUT_DIR = PROJECT_ROOT / "runs" / "tensorrt_mask_overlays"
DEFAULT_OVERLAP_OUTPUT_DIR = PROJECT_ROOT / "runs" / "tensorrt_overlap_visualizations"
DEFAULT_CLEARANCE_OUTPUT_DIR = PROJECT_ROOT / "runs" / "stent_clearance_checks"
DEFAULT_RUNTIME_CONFIG = PROJECT_ROOT / "runtime_config.yaml"

WARMUP_RUNS = 10
MASK_CONFIDENCE_THRESHOLD = 0.25
MASK_THRESHOLD = 0.5
MASK_ALPHA = 0.45
MAX_MASKS_PER_IMAGE = 20

DORN_CLASS_NAME = "Dorn"
STENT_CLEARANCE_CLASS_NAME = "StentClearance"
TARGET_CLASSES = (DORN_CLASS_NAME, STENT_CLEARANCE_CLASS_NAME)

MAX_DORN_OUTSIDE_CLEARANCE_PERCENT = 1.0
MIN_CLEARANCE_GAP_PX = 3
MAX_CENTER_OFFSET_RATIO = 0.25
MIN_CLEARANCE_AREA_FACTOR = 1.15

CLEARANCE_CHECK_ALPHA = 0.55
GAP_FORBIDDEN_ALPHA = 0.55
CENTER_FORBIDDEN_ALPHA = 0.28


def load_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        return {}
    data = yaml.safe_load(data_yaml.read_text()) or {}
    names = data.get("names", {})
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    if isinstance(names, list):
        return {class_id: str(name) for class_id, name in enumerate(names)}
    return {}


CLASS_NAMES = load_class_names(DATA_YAML)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def collect_images(input_path: str | Path) -> list[Path]:
    path = resolve_path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path must be an image file or directory: {path}")
    images = sorted(
        (p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not images:
        raise ValueError(f"No supported images found under: {path}")
    return images
