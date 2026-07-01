from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MODEL_H, MODEL_W

_CV2 = None
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_INV_STD = np.array([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=np.float32)
_INPUT_SCALE = np.float32(1.0 / 255.0)


def require_cv2():
    global _CV2
    if _CV2 is None:
        import cv2

        _CV2 = cv2
    return _CV2


def normalize_rgb_array(image_rgb: np.ndarray) -> np.ndarray:
    array = image_rgb.astype(np.float32, copy=False)
    array = (array * _INPUT_SCALE - _RGB_MEAN) * _RGB_INV_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...])


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
    return normalize_rgb_array(np.asarray(image))


def preprocess_path(image_path: Path) -> np.ndarray:
    cv2 = require_cv2()
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is not None:
        return preprocess_bgr_frame(frame)
    with Image.open(image_path) as image:
        return preprocess_image(image)


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if frame_bgr.shape[:2] == (MODEL_H, MODEL_W):
        resized = frame_bgr
    else:
        resized = cv2.resize(frame_bgr, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
    output = np.empty((1, 3, MODEL_H, MODEL_W), dtype=np.float32)
    output[0, 0] = (resized[:, :, 2].astype(np.float32) * _INPUT_SCALE - _RGB_MEAN[0]) * _RGB_INV_STD[0]
    output[0, 1] = (resized[:, :, 1].astype(np.float32) * _INPUT_SCALE - _RGB_MEAN[1]) * _RGB_INV_STD[1]
    output[0, 2] = (resized[:, :, 0].astype(np.float32) * _INPUT_SCALE - _RGB_MEAN[2]) * _RGB_INV_STD[2]
    return output


def parse_camera_source(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "false"}:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop-box must be four comma-separated values: x1,y1,x2,y2")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--crop-box values must be integers") from exc


def configure_capture(
    cap,
    capture_width: int,
    capture_height: int,
    capture_fps: float = 0.0,
    capture_fourcc: str | None = None,
) -> None:
    cv2 = require_cv2()
    fourcc = str(capture_fourcc or "").strip()
    if fourcc:
        if len(fourcc) != 4:
            raise ValueError("capture_fourcc must be exactly four characters, for example MJPG or YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if capture_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
    if capture_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
    if capture_fps and capture_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(capture_fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def crop_camera_frame(frame_bgr: np.ndarray, crop_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop_box is None:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = crop_box
    x1, x2 = max(0, min(width, int(x1))), max(0, min(width, int(x2)))
    y1, y2 = max(0, min(height, int(y1))), max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]
