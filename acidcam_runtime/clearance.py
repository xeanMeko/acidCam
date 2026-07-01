from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    CLASS_NAMES,
    DORN_CLASS_NAME,
    MASK_CONFIDENCE_THRESHOLD,
    MASK_THRESHOLD,
    MAX_CENTER_OFFSET_RATIO,
    MAX_DORN_OUTSIDE_CLEARANCE_PERCENT,
    MIN_CLEARANCE_AREA_FACTOR,
    MIN_CLEARANCE_GAP_PX,
    STENT_CLEARANCE_CLASS_NAME,
    TARGET_CLASSES,
)
from .postprocess import (
    binary_mask_stats,
    class_predictions,
    find_output,
    mask_probabilities,
    resize_mask,
    selected_mask_records,
    squeeze_batch,
)

_TARGET_CANDIDATE_LIMIT = 4


def normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def class_name_matches(name: str, target_name: str) -> bool:
    name = normalize_class_name(name)
    target = normalize_class_name(target_name)
    return bool(name and target and (name == target or target in name or name in target))


def find_best_class_record(records: list[dict[str, Any]], target_name: str, required: bool = True) -> dict[str, Any] | None:
    matches = [record for record in records if class_name_matches(record.get("class_name", ""), target_name)]
    if not matches:
        if required:
            available = sorted({record.get("class_name", "unknown") for record in records})
            raise ValueError(f"Could not find selected mask for class {target_name!r}. Available selected classes: {available}")
        return None
    return max(matches, key=lambda record: (float(record.get("score", 0.0)), int(record.get("area_pixels", 0))))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return binary_mask_stats(mask)[1]


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    return binary_mask_stats(mask)[2]


def record_bbox(record: dict[str, Any], mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = record.get("bbox")
    return bbox if bbox is not None else mask_bbox(mask)


def record_centroid(record: dict[str, Any], mask: np.ndarray) -> tuple[float, float] | None:
    centroid = record.get("centroid")
    return centroid if centroid is not None else mask_centroid(mask)


def bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return x2 - x1, y2 - y1


def bbox_gaps(inner_bbox: tuple[int, int, int, int], outer_bbox: tuple[int, int, int, int]) -> dict[str, int]:
    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    return {"left": ix1 - ox1, "right": ox2 - ix2, "top": iy1 - oy1, "bottom": oy2 - iy2}


def target_for_class_id(class_id: int) -> tuple[str, str | None]:
    class_name = CLASS_NAMES.get(class_id, str(class_id))
    target_name = next((target for target in TARGET_CLASSES if class_name_matches(class_name, target)), None)
    return class_name, target_name


def class_only_presence_precheck(output_map: dict[str, np.ndarray]) -> dict[str, Any] | None:
    masks = find_output(output_map, "masks", "mask")
    if masks is None:
        return None
    masks = squeeze_batch(masks)
    if masks.ndim == 2:
        query_count = 1
    elif masks.ndim == 3:
        query_count = int(masks.shape[0])
    else:
        return None

    scores, class_ids = class_predictions(output_map, query_count)
    selected = np.where(scores >= float(MASK_CONFIDENCE_THRESHOLD))[0]
    available_classes: set[str] = set()
    present_targets: set[str] = set()

    for mask_index in selected:
        class_id = int(class_ids[mask_index]) if mask_index < len(class_ids) else 0
        class_name, target_name = target_for_class_id(class_id)
        available_classes.add(class_name)
        if target_name is not None:
            present_targets.add(target_name)

    missing_classes = [target for target in TARGET_CLASSES if target not in present_targets]
    if not missing_classes:
        return None

    return {
        "missing_classes": missing_classes,
        "available_classes": sorted(available_classes),
    }


def stent_clearance_records_for_frame(output_map: dict[str, np.ndarray], frame_shape: tuple[int, ...]) -> list[dict[str, Any]]:
    masks = find_output(output_map, "masks", "mask")
    if masks is None:
        return []
    masks = squeeze_batch(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        return []
    scores, class_ids = class_predictions(output_map, masks.shape[0])
    selected = np.where(scores >= float(MASK_CONFIDENCE_THRESHOLD))[0]
    if selected.size == 0:
        return []
    selected = selected[np.argsort(scores[selected])[::-1]]

    candidates_by_target: dict[str, list[tuple[int, int, int, str, float]]] = {}
    for rank, mask_index in enumerate(selected):
        if all(len(candidates_by_target.get(target, [])) >= _TARGET_CANDIDATE_LIMIT for target in TARGET_CLASSES):
            break
        score = float(scores[mask_index]) if mask_index < len(scores) else 1.0
        class_id = int(class_ids[mask_index]) if mask_index < len(class_ids) else 0
        class_name, target_name = target_for_class_id(class_id)
        if target_name is None:
            continue
        candidates = candidates_by_target.setdefault(target_name, [])
        if len(candidates) < _TARGET_CANDIDATE_LIMIT:
            candidates.append((rank, int(mask_index), class_id, class_name, score))

    frame_height, frame_width = frame_shape[:2]
    records_by_target: dict[str, dict[str, Any]] = {}
    for target_name in TARGET_CLASSES:
        for rank, mask_index, class_id, class_name, score in candidates_by_target.get(target_name, []):
            previous = records_by_target.get(target_name)
            if previous is not None and score < float(previous["score"]):
                break
            mask_prob = mask_probabilities(masks[mask_index])
            binary = resize_mask(mask_prob, (frame_width, frame_height)) >= float(MASK_THRESHOLD)
            area, bbox, centroid = binary_mask_stats(binary)
            if area == 0:
                continue
            record = {
                "rank": rank,
                "query_index": int(mask_index),
                "class_id": class_id,
                "class_name": class_name,
                "target_name": target_name,
                "score": score,
                "area_pixels": area,
                "bbox": bbox,
                "centroid": centroid,
                "mask": binary,
            }
            if previous is None or (record["score"], record["area_pixels"]) > (previous["score"], previous["area_pixels"]):
                records_by_target[target_name] = record
    records = [records_by_target[target] for target in TARGET_CLASSES if target in records_by_target]
    for rank, record in enumerate(records):
        record["rank"] = rank
    return records


def clearance_mask_records(output_map: dict[str, np.ndarray], image_path: Path | None = None) -> list[dict[str, Any]]:
    frame_shape = output_map.get("_frame_shape") if isinstance(output_map, dict) else None
    if frame_shape is not None:
        return stent_clearance_records_for_frame(output_map, frame_shape)
    return selected_mask_records(output_map, image_path)


def analysis_mask_shape(records: list[dict[str, Any]], image_path: Path | None, output_map: dict[str, np.ndarray]) -> tuple[int, int]:
    if records:
        return tuple(records[0]["mask"].shape)
    frame_shape = output_map.get("_frame_shape") if isinstance(output_map, dict) else None
    if frame_shape is not None:
        return int(frame_shape[0]), int(frame_shape[1])
    if image_path is not None:
        width, height = Image.open(image_path).size
        return height, width
    masks = find_output(output_map, "masks", "mask")
    if masks is not None:
        masks = squeeze_batch(masks)
        if masks.ndim >= 2:
            return tuple(masks.shape[-2:])
    return 1, 1


def empty_fit_report(
    output_map: dict[str, np.ndarray],
    image_path: Path | None,
    records: list[dict[str, Any]] | None = None,
    dorn: dict[str, Any] | None = None,
    clearance: dict[str, Any] | None = None,
    reasons: list[str] | None = None,
    missing_classes: list[str] | None = None,
    available_classes: list[str] | None = None,
    postprocess_skipped: bool = False,
) -> dict[str, Any]:
    records = records or []
    reasons = reasons or []
    height, width = analysis_mask_shape(records, image_path, output_map)
    empty = np.zeros((height, width), dtype=bool)
    dorn_mask = dorn["mask"].astype(bool) if dorn is not None else empty.copy()
    clearance_mask = clearance["mask"].astype(bool) if clearance is not None else empty.copy()
    missing = list(missing_classes) if missing_classes is not None else []
    if missing_classes is None:
        if dorn is None:
            missing.append(DORN_CLASS_NAME)
        if clearance is None:
            missing.append(STENT_CLEARANCE_CLASS_NAME)
    available = available_classes
    if available is None:
        available = sorted({record.get("class_name", "unknown") for record in records})
    dorn_area = int(dorn.get("area_pixels") if dorn is not None and dorn.get("area_pixels") is not None else np.count_nonzero(dorn_mask))
    clearance_area = int(clearance.get("area_pixels") if clearance is not None and clearance.get("area_pixels") is not None else np.count_nonzero(clearance_mask))
    inside_pixels = int((dorn_mask & clearance_mask).sum()) if dorn is not None and clearance is not None else 0
    outside_pixels = int((dorn_mask & ~clearance_mask).sum()) if dorn is not None and clearance is not None else dorn_area
    return {
        "pass_clearance": False,
        "status": "FAIL",
        "reasons": reasons,
        "missing_classes": missing,
        "available_classes": available,
        "postprocess_skipped": bool(postprocess_skipped),
        "dorn": dorn,
        "clearance": clearance,
        "dorn_mask": dorn_mask,
        "clearance_mask": clearance_mask,
        "outside_mask": np.zeros_like(dorn_mask, dtype=bool),
        "inside_percent": 100.0 * inside_pixels / dorn_area if dorn_area and clearance is not None else 0.0,
        "outside_percent": 100.0 * outside_pixels / dorn_area if dorn_area else (100.0 if dorn is None else 0.0),
        "clearance_occupied_percent": 100.0 * inside_pixels / clearance_area if clearance_area and dorn is not None else 0.0,
        "area_factor": clearance_area / dorn_area if dorn_area and clearance_area else 0.0,
        "dorn_area": dorn_area,
        "clearance_area": clearance_area,
        "dorn_bbox": record_bbox(dorn, dorn_mask) if dorn is not None else mask_bbox(dorn_mask),
        "clearance_bbox": record_bbox(clearance, clearance_mask) if clearance is not None else mask_bbox(clearance_mask),
        "gaps": None,
        "min_gap_px": None,
        "dorn_center": record_centroid(dorn, dorn_mask) if dorn is not None else mask_centroid(dorn_mask),
        "clearance_center": record_centroid(clearance, clearance_mask) if clearance is not None else mask_centroid(clearance_mask),
        "center_dx": None,
        "center_dy": None,
        "center_offset_px": None,
        "center_limit_px": None,
    }


def analyze_stent_clearance_fit(
    output_map: dict[str, np.ndarray],
    image_path: Path | None = None,
    skip_missing_class_postprocess: bool = True,
) -> dict[str, Any]:
    if skip_missing_class_postprocess:
        try:
            precheck = class_only_presence_precheck(output_map)
        except Exception:
            precheck = None
        if precheck is not None:
            missing_classes = precheck["missing_classes"]
            reasons = [f"{class_name} class is missing" for class_name in missing_classes]
            return empty_fit_report(
                output_map,
                image_path,
                records=[],
                reasons=reasons,
                missing_classes=missing_classes,
                available_classes=precheck["available_classes"],
                postprocess_skipped=True,
            )

    try:
        records = clearance_mask_records(output_map, image_path)
    except Exception as exc:
        return empty_fit_report(output_map, image_path, records=[], reasons=[f"Could not read predicted masks: {type(exc).__name__}: {exc}"])
    dorn = find_best_class_record(records, DORN_CLASS_NAME, required=False)
    clearance = find_best_class_record(records, STENT_CLEARANCE_CLASS_NAME, required=False)
    reasons = []
    if dorn is None:
        reasons.append(f"{DORN_CLASS_NAME} mask is missing")
        return empty_fit_report(output_map, image_path, records=records, dorn=dorn, clearance=clearance, reasons=reasons)
    if clearance is None:
        reasons.append(f"{STENT_CLEARANCE_CLASS_NAME} mask is missing")
        return empty_fit_report(output_map, image_path, records=records, dorn=dorn, clearance=clearance, reasons=reasons)
    if reasons:
        return empty_fit_report(output_map, image_path, records=records, dorn=dorn, clearance=clearance, reasons=reasons)

    dorn_mask = dorn["mask"].astype(bool, copy=False)
    clearance_mask = clearance["mask"].astype(bool, copy=False)
    dorn_area = int(dorn.get("area_pixels") or np.count_nonzero(dorn_mask))
    clearance_area = int(clearance.get("area_pixels") or np.count_nonzero(clearance_mask))
    outside_mask = dorn_mask & ~clearance_mask
    outside_pixels = int(np.count_nonzero(outside_mask))
    inside_pixels = dorn_area - outside_pixels
    dorn_bbox = record_bbox(dorn, dorn_mask)
    clearance_bbox = record_bbox(clearance, clearance_mask)
    if dorn_bbox is None or clearance_bbox is None:
        reasons = []
        if dorn_bbox is None:
            reasons.append(f"{DORN_CLASS_NAME} mask contains no pixels after thresholding")
        if clearance_bbox is None:
            reasons.append(f"{STENT_CLEARANCE_CLASS_NAME} mask contains no pixels after thresholding")
        return empty_fit_report(output_map, image_path, records=records, dorn=dorn, clearance=clearance, reasons=reasons)

    gaps = bbox_gaps(dorn_bbox, clearance_bbox)
    min_gap_px = min(gaps.values())
    clearance_width, clearance_height = bbox_size(clearance_bbox)
    dorn_center = record_centroid(dorn, dorn_mask)
    clearance_center = record_centroid(clearance, clearance_mask)
    center_dx = float(dorn_center[0] - clearance_center[0])
    center_dy = float(dorn_center[1] - clearance_center[1])
    center_offset_px = float((center_dx**2 + center_dy**2) ** 0.5)
    center_limit_px = float(min(clearance_width, clearance_height) * MAX_CENTER_OFFSET_RATIO)
    outside_percent = 100.0 * outside_pixels / dorn_area if dorn_area else 100.0
    inside_percent = 100.0 * inside_pixels / dorn_area if dorn_area else 0.0
    clearance_occupied_percent = 100.0 * inside_pixels / clearance_area if clearance_area else 100.0
    area_factor = clearance_area / dorn_area if dorn_area else 0.0

    containment_ok = outside_percent <= float(MAX_DORN_OUTSIDE_CLEARANCE_PERCENT)
    gap_ok = min_gap_px >= int(MIN_CLEARANCE_GAP_PX)
    center_ok = center_offset_px <= center_limit_px
    area_ok = area_factor >= float(MIN_CLEARANCE_AREA_FACTOR)
    pass_clearance = containment_ok and gap_ok and center_ok and area_ok
    reasons = []
    if not containment_ok:
        reasons.append(f"{outside_percent:.2f}% of Dorn mask is outside the clearance")
    if not gap_ok:
        reasons.append(f"minimum bbox gap is {min_gap_px}px, below {MIN_CLEARANCE_GAP_PX}px")
    if not center_ok:
        reasons.append(f"center offset is {center_offset_px:.1f}px, above {center_limit_px:.1f}px")
    if not area_ok:
        reasons.append(f"clearance area factor is {area_factor:.2f}x, below {MIN_CLEARANCE_AREA_FACTOR:.2f}x")
    return {
        "pass_clearance": pass_clearance,
        "status": "PASS" if pass_clearance else "FAIL",
        "reasons": reasons,
        "missing_classes": [],
        "available_classes": sorted({record.get("class_name", "unknown") for record in records}),
        "postprocess_skipped": False,
        "dorn": dorn,
        "clearance": clearance,
        "dorn_mask": dorn_mask,
        "clearance_mask": clearance_mask,
        "outside_mask": outside_mask,
        "inside_percent": inside_percent,
        "outside_percent": outside_percent,
        "clearance_occupied_percent": clearance_occupied_percent,
        "area_factor": area_factor,
        "dorn_area": dorn_area,
        "clearance_area": clearance_area,
        "dorn_bbox": dorn_bbox,
        "clearance_bbox": clearance_bbox,
        "gaps": gaps,
        "min_gap_px": min_gap_px,
        "dorn_center": dorn_center,
        "clearance_center": clearance_center,
        "center_dx": center_dx,
        "center_dy": center_dy,
        "center_offset_px": center_offset_px,
        "center_limit_px": center_limit_px,
    }


def format_metric(value, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}{suffix}"
    return f"{value}{suffix}"


def frame_output_map_for_clearance(output_map: dict[str, np.ndarray], frame_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    live_output_map = dict(output_map)
    live_output_map["_frame_shape"] = tuple(int(value) for value in frame_shape)
    return live_output_map


def synthetic_output_map(dorn_mask: np.ndarray | None, clearance_mask: np.ndarray | None) -> dict[str, np.ndarray]:
    masks = []
    labels = []
    if dorn_mask is not None:
        masks.append(dorn_mask.astype(np.float32))
        labels.append([0.01, 0.98, 0.01])
    if clearance_mask is not None:
        masks.append(clearance_mask.astype(np.float32))
        labels.append([0.01, 0.01, 0.98])
    if not masks:
        masks.append(np.zeros((64, 64), dtype=np.float32))
        labels.append([0.98, 0.01, 0.01])
    return {
        "masks": np.asarray(masks, dtype=np.float32)[None, ...],
        "labels": np.asarray(labels, dtype=np.float32)[None, ...],
        "_frame_shape": (masks[0].shape[0], masks[0].shape[1], 3),
    }


def run_clearance_smoke_tests() -> int:
    clearance = np.zeros((64, 64), dtype=bool)
    clearance[10:54, 10:54] = True
    dorn_pass = np.zeros((64, 64), dtype=bool)
    dorn_pass[22:42, 22:42] = True
    dorn_fail = np.zeros((64, 64), dtype=bool)
    dorn_fail[4:44, 4:44] = True
    pass_report = analyze_stent_clearance_fit(synthetic_output_map(dorn_pass, clearance))
    fail_report = analyze_stent_clearance_fit(synthetic_output_map(dorn_fail, clearance))
    missing_report = analyze_stent_clearance_fit(synthetic_output_map(dorn_pass, None))
    missing_dorn_report = analyze_stent_clearance_fit(synthetic_output_map(None, clearance))
    missing_no_skip_report = analyze_stent_clearance_fit(
        synthetic_output_map(dorn_pass, None),
        skip_missing_class_postprocess=False,
    )
    assert pass_report["pass_clearance"], pass_report
    assert not pass_report["postprocess_skipped"], pass_report
    assert not fail_report["pass_clearance"], fail_report
    assert not fail_report["postprocess_skipped"], fail_report
    assert (
        not missing_report["pass_clearance"]
        and missing_report["missing_classes"] == [STENT_CLEARANCE_CLASS_NAME]
        and missing_report["postprocess_skipped"]
    ), missing_report
    assert (
        not missing_dorn_report["pass_clearance"]
        and missing_dorn_report["missing_classes"] == [DORN_CLASS_NAME]
        and missing_dorn_report["postprocess_skipped"]
    ), missing_dorn_report
    assert (
        not missing_no_skip_report["pass_clearance"]
        and missing_no_skip_report["missing_classes"] == [STENT_CLEARANCE_CLASS_NAME]
        and not missing_no_skip_report["postprocess_skipped"]
        and missing_no_skip_report["dorn"] is not None
    ), missing_no_skip_report
    print("Clearance smoke tests passed.")
    return 0
