from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .preprocess import require_cv2

from .config import (
    CLASS_NAMES,
    MASK_ALPHA,
    MASK_CONFIDENCE_THRESHOLD,
    MASK_THRESHOLD,
    MAX_MASKS_PER_IMAGE,
    resolve_path,
)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("._") or "output"


def tensor_stats(name: str, array: np.ndarray) -> str:
    prefix = f"{name}: shape={array.shape}, dtype={array.dtype}"
    if array.size == 0:
        return f"{prefix}, empty"
    if np.issubdtype(array.dtype, np.number):
        return f"{prefix}, min={array.min():.6g}, max={array.max():.6g}, mean={array.mean():.6g}"
    return prefix


def find_output(output_map: dict[str, np.ndarray], *name_parts: str) -> np.ndarray | None:
    lowered = {name.lower(): value for name, value in output_map.items() if not name.startswith("_")}
    for name_part in name_parts:
        if name_part.lower() in lowered:
            return lowered[name_part.lower()]
    for name, value in output_map.items():
        if name.startswith("_"):
            continue
        if any(name_part.lower() in name.lower() for name_part in name_parts):
            return value
    return None


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32, copy=False), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    values = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def squeeze_batch(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim >= 1 and array.shape[0] == 1:
        return array[0]
    return array


def class_predictions(output_map: dict[str, np.ndarray], query_count: int) -> tuple[np.ndarray, np.ndarray]:
    labels = find_output(output_map, "labels", "scores", "logits")
    if labels is None:
        return np.ones(query_count, dtype=np.float32), np.zeros(query_count, dtype=np.int64)
    labels = squeeze_batch(labels).astype(np.float32, copy=False)
    if labels.ndim == 1:
        scores = labels[:query_count]
        return scores, np.zeros_like(scores, dtype=np.int64)
    labels = labels[:query_count]
    if labels.size == 0:
        return np.zeros(query_count, dtype=np.float32), np.zeros(query_count, dtype=np.int64)
    row_sums = labels.sum(axis=-1)
    looks_like_probs = labels.min() >= 0.0 and labels.max() <= 1.0 and np.all(row_sums <= 1.05)
    probabilities = labels if looks_like_probs else softmax(labels, axis=-1)
    class_count = len(CLASS_NAMES)
    if class_count and probabilities.shape[-1] == class_count + 1:
        probabilities = probabilities[:, :class_count]
    elif class_count and probabilities.shape[-1] > class_count:
        probabilities = probabilities[:, :class_count]
    class_ids = np.argmax(probabilities, axis=-1).astype(np.int64)
    scores = probabilities[np.arange(probabilities.shape[0]), class_ids]
    return scores.astype(np.float32), class_ids


def mask_probabilities(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if not np.issubdtype(masks.dtype, np.floating):
        masks = masks.astype(np.float32, copy=False)
    if masks.size and (float(masks.min()) < 0.0 or float(masks.max()) > 1.0):
        masks = sigmoid(masks)
    return masks


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if tuple(mask.shape[::-1]) == tuple(size):
        return mask
    cv2 = require_cv2()
    return cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)


def binary_mask_stats(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int] | None, tuple[float, float] | None]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0, None, None
    area = int(xs.size)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    centroid = (float(xs.mean()), float(ys.mean()))
    return area, bbox, centroid


def color_for_index(index: int) -> np.ndarray:
    palette = np.array(
        [
            [230, 57, 70],
            [42, 157, 143],
            [38, 70, 83],
            [244, 162, 97],
            [69, 123, 157],
            [131, 56, 236],
            [255, 183, 3],
            [0, 150, 199],
        ],
        dtype=np.float32,
    )
    return palette[index % len(palette)]


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int]) -> None:
    x, y = xy
    try:
        bbox = draw.textbbox((x, y), text)
    except AttributeError:
        w, h = draw.textsize(text)
        bbox = (x, y, x + w, y + h)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=color)


def selected_mask_records(output_map: dict[str, np.ndarray], image_path: Path | None = None) -> list[dict[str, Any]]:
    masks = find_output(output_map, "masks", "mask")
    if masks is None:
        raise KeyError(f"No masks output found. Available outputs: {list(output_map)}")
    masks = squeeze_batch(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (queries, height, width), got {masks.shape}")
    masks = mask_probabilities(masks)
    scores, class_ids = class_predictions(output_map, masks.shape[0])
    selected = np.where(scores >= float(MASK_CONFIDENCE_THRESHOLD))[0]
    selected = selected[np.argsort(scores[selected])[::-1]] if selected.size else selected
    selected = selected[: int(MAX_MASKS_PER_IMAGE)] if MAX_MASKS_PER_IMAGE is not None else selected

    image_size = None
    if image_path is not None:
        image_size = Image.open(image_path).size
    elif output_map.get("_frame_shape") is not None:
        frame_shape = output_map["_frame_shape"]
        image_size = (int(frame_shape[1]), int(frame_shape[0]))

    records = []
    for rank, mask_index in enumerate(selected):
        mask_prob = masks[mask_index]
        if image_size is not None:
            mask_prob = resize_mask(mask_prob, image_size)
        binary = mask_prob >= float(MASK_THRESHOLD)
        area, bbox, centroid = binary_mask_stats(binary)
        if area == 0:
            continue
        class_id = int(class_ids[mask_index]) if mask_index < len(class_ids) else 0
        score = float(scores[mask_index]) if mask_index < len(scores) else 1.0
        records.append(
            {
                "rank": rank,
                "query_index": int(mask_index),
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "score": score,
                "area_pixels": area,
                "bbox": bbox,
                "centroid": centroid,
                "mask": binary,
            }
        )
    return records


def save_raw_outputs(image_path: Path, output_map: dict[str, np.ndarray], output_dir: str | Path) -> Path:
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {safe_name(name): value for name, value in output_map.items() if not name.startswith("_")}
    output_path = output_dir / f"{safe_name(image_path.stem)}.npz"
    np.savez_compressed(output_path, **payload)
    return output_path


def save_mask_overlay(image_path: Path, output_map: dict[str, np.ndarray], output_dir: str | Path) -> tuple[Path, int]:
    records = selected_mask_records(output_map, image_path)
    image = Image.open(image_path).convert("RGB")
    overlay = np.asarray(image, dtype=np.float32).copy()
    labels = []
    for rank, record in enumerate(records):
        color = color_for_index(record["class_id"] if CLASS_NAMES else rank)
        binary = record["mask"]
        overlay[binary] = (1.0 - MASK_ALPHA) * overlay[binary] + MASK_ALPHA * color
        ys, xs = np.where(binary)
        if xs.size:
            labels.append(
                (
                    int(xs.mean()),
                    max(0, int(ys.min()) - 14),
                    f"{record['class_name']} {record['score']:.2f}",
                    tuple(color.astype(int)),
                )
            )
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for x, y, text, color in labels:
        draw_label(draw, (x, y), text, color)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(image_path.stem)}_masks.png"
    out.save(output_path)
    return output_path, len(labels)
