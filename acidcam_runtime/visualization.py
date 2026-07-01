from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .clearance import (
    format_metric,
    mask_bbox,
    mask_centroid,
)
from .config import (
    CENTER_FORBIDDEN_ALPHA,
    CLEARANCE_CHECK_ALPHA,
    DEFAULT_CLEARANCE_OUTPUT_DIR,
    GAP_FORBIDDEN_ALPHA,
    MAX_CENTER_OFFSET_RATIO,
    MIN_CLEARANCE_GAP_PX,
    resolve_path,
)
from .postprocess import safe_name, selected_mask_records
from .preprocess import require_cv2


def mask_overlap_summary(output_map: dict[str, np.ndarray], image_path: Path | None = None) -> dict[str, Any]:
    records = selected_mask_records(output_map, image_path)
    if not records:
        return {"mask_count": 0, "union_pixels": 0, "overlap_pixels": 0, "overall_overlap_percent": 0.0, "repeated_mask_area_percent": 0.0, "per_mask": [], "pairwise": []}
    stack = np.stack([record["mask"] for record in records], axis=0).astype(bool)
    coverage = stack.sum(axis=0)
    union_pixels = int((coverage >= 1).sum())
    overlap_pixels = int((coverage >= 2).sum())
    total_mask_pixels = int(stack.sum())
    repeated_pixels = total_mask_pixels - union_pixels
    per_mask = []
    for index, record in enumerate(records):
        mask = stack[index]
        other_coverage = np.delete(stack, index, axis=0).any(axis=0) if len(records) > 1 else np.zeros_like(mask)
        mask_overlap_pixels = int((mask & other_coverage).sum())
        mask_area = int(mask.sum())
        per_mask.append(
            {
                **{key: record[key] for key in ("rank", "query_index", "class_id", "class_name", "score", "area_pixels")},
                "overlap_pixels": mask_overlap_pixels,
                "overlap_percent_of_mask": 100.0 * mask_overlap_pixels / mask_area if mask_area else 0.0,
            }
        )
    pairwise = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            left_mask = stack[left]
            right_mask = stack[right]
            intersection = int((left_mask & right_mask).sum())
            if intersection == 0:
                continue
            pair_union = int((left_mask | right_mask).sum())
            smaller_area = min(int(left_mask.sum()), int(right_mask.sum()))
            pairwise.append(
                {
                    "left_rank": records[left]["rank"],
                    "right_rank": records[right]["rank"],
                    "left_name": records[left]["class_name"],
                    "right_name": records[right]["class_name"],
                    "intersection_pixels": intersection,
                    "iou_percent": 100.0 * intersection / pair_union if pair_union else 0.0,
                    "percent_of_smaller_mask": 100.0 * intersection / smaller_area if smaller_area else 0.0,
                }
            )
    pairwise.sort(key=lambda row: row["percent_of_smaller_mask"], reverse=True)
    return {
        "mask_count": len(records),
        "union_pixels": union_pixels,
        "overlap_pixels": overlap_pixels,
        "overall_overlap_percent": 100.0 * overlap_pixels / union_pixels if union_pixels else 0.0,
        "repeated_mask_area_percent": 100.0 * repeated_pixels / total_mask_pixels if total_mask_pixels else 0.0,
        "per_mask": per_mask,
        "pairwise": pairwise,
    }


def overlap_color_overlay(base_rgb: np.ndarray, coverage: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    overlay = base_rgb.astype(np.float32).copy()
    for mask, color in (
        (coverage == 1, np.array([42, 157, 143], dtype=np.float32)),
        (coverage == 2, np.array([255, 183, 3], dtype=np.float32)),
        (coverage >= 3, np.array([230, 57, 70], dtype=np.float32)),
    ):
        if np.any(mask):
            overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_overlap_legend(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    draw.text((x, y), "Overlap coverage", fill=(255, 255, 255))
    y += 22
    for color, label in [((42, 157, 143), "1 mask"), ((255, 183, 3), "2 masks overlap"), ((230, 57, 70), "3+ masks overlap")]:
        draw.rectangle((x, y, x + 18, y + 18), fill=color)
        draw.text((x + 26, y), label, fill=(255, 255, 255))
        y += 24


def visualize_mask_overlap(output_map: dict[str, np.ndarray], image_path: Path, output_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    image_path = resolve_path(image_path)
    image = Image.open(image_path).convert("RGB")
    base = np.asarray(image, dtype=np.uint8)
    records = selected_mask_records(output_map, image_path)
    coverage = np.stack([r["mask"] for r in records], axis=0).sum(axis=0) if records else np.zeros(base.shape[:2], dtype=np.int32)
    overlay_image = Image.fromarray(overlap_color_overlay(base, coverage))
    gap = 12
    legend_height = 92
    canvas = Image.new("RGB", (image.width * 2 + gap, image.height + legend_height), (20, 20, 20))
    canvas.paste(image, (0, 0))
    canvas.paste(overlay_image, (image.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Original", fill=(255, 255, 255))
    draw.text((image.width + gap + 8, 8), "Mask overlap", fill=(255, 255, 255))
    draw_overlap_legend(draw, 12, image.height + 12)
    summary = mask_overlap_summary(output_map, image_path)
    text = f"Selected masks: {summary['mask_count']} | 2+ coverage: {summary['overall_overlap_percent']:.2f}% | repeated: {summary['repeated_mask_area_percent']:.2f}%"
    draw.text((image.width + gap + 12, image.height + 12), text, fill=(255, 255, 255))
    if canvas.width > 1800:
        scale = 1800 / canvas.width
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.BILINEAR)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(image_path.stem)}_overlap.png"
    canvas.save(output_path)
    return output_path, summary


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int] | None, color: tuple[int, int, int], label: str) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    draw.text((x1 + 4, max(0, y1 - 16)), label, fill=color)


def draw_center(draw: ImageDraw.ImageDraw, center: tuple[float, float] | None, color: tuple[int, int, int]) -> None:
    if center is None:
        return
    x, y = (int(round(value)) for value in center)
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def clipped_bbox_for_shape(bbox: tuple[int, int, int, int] | None, shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_region_mask(bbox: tuple[int, int, int, int] | None, shape: tuple[int, ...]) -> np.ndarray | None:
    clipped = clipped_bbox_for_shape(bbox, shape)
    if clipped is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    x1, y1, x2, y2 = clipped
    mask[y1:y2, x1:x2] = True
    return mask


def clearance_gap_forbidden_mask(report: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray | None:
    bbox = clipped_bbox_for_shape(report.get("clearance_bbox"), shape)
    if bbox is None:
        return None
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    gap = max(0, int(MIN_CLEARANCE_GAP_PX))
    if gap == 0:
        return mask
    x1, y1, x2, y2 = bbox
    mask[y1 : min(y2, y1 + gap), x1:x2] = True
    mask[max(y1, y2 - gap) : y2, x1:x2] = True
    mask[y1:y2, x1 : min(x2, x1 + gap)] = True
    mask[y1:y2, max(x1, x2 - gap) : x2] = True
    return mask


def center_offset_allowed_mask(report: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray | None:
    center = report.get("clearance_center")
    limit = report.get("center_limit_px")
    bbox = clipped_bbox_for_shape(report.get("clearance_bbox"), shape)
    if center is None or limit is None or bbox is None or float(limit) <= 0:
        return None
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    cx, cy = (float(value) for value in center)
    x1, y1, x2, y2 = bbox
    ys, xs = np.ogrid[y1:y2, x1:x2]
    mask[y1:y2, x1:x2] = (xs - cx) ** 2 + (ys - cy) ** 2 <= float(limit) ** 2
    return mask


def center_offset_forbidden_mask(report: dict[str, Any], shape: tuple[int, ...]) -> np.ndarray | None:
    region = bbox_region_mask(report.get("clearance_bbox"), shape)
    allowed = center_offset_allowed_mask(report, shape)
    if region is None or allowed is None:
        return None
    return region & ~allowed


def blend_mask_overlay(overlay: np.ndarray, mask: np.ndarray | None, color: np.ndarray, alpha: float) -> None:
    if mask is not None and np.any(mask):
        overlay[mask] = (1.0 - float(alpha)) * overlay[mask] + float(alpha) * color


def draw_clearance_threshold_guides(draw: ImageDraw.ImageDraw, report: dict[str, Any]) -> None:
    bbox = report.get("clearance_bbox")
    gap = max(0, int(MIN_CLEARANCE_GAP_PX))
    if bbox is not None and gap > 0:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        safe_bbox = (x1 + gap, y1 + gap, x2 - gap, y2 - gap)
        if safe_bbox[2] > safe_bbox[0] and safe_bbox[3] > safe_bbox[1]:
            draw.rectangle(safe_bbox, outline=(255, 183, 3), width=2)
            draw.text((safe_bbox[0] + 4, max(0, safe_bbox[1] + 4)), f"gap >= {gap}px", fill=(255, 183, 3))
    center = report.get("clearance_center")
    limit = report.get("center_limit_px")
    if center is not None and limit is not None and float(limit) > 0:
        cx, cy = (float(value) for value in center)
        radius = float(limit)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(255, 120, 255), width=3)


def clearance_footer_lines(report: dict[str, Any]) -> list[str]:
    lines = [f"Stent clearance check: {report.get('status', 'n/a')}"]
    if report.get("missing_classes"):
        lines.append(f"Missing: {', '.join(report['missing_classes'])}")
        lines.append(f"Visible classes: {', '.join(report.get('available_classes', [])) or 'none'}")
    else:
        lines.append(
            f"Out {format_metric(report.get('outside_percent'), '%', 1)} | "
            f"gap {format_metric(report.get('min_gap_px'))}/{MIN_CLEARANCE_GAP_PX}px | "
            f"off {format_metric(report.get('center_offset_px'), 'px', 1)}/{format_metric(report.get('center_limit_px'), 'px', 1)}"
        )
        lines.append(f"Area {format_metric(report.get('area_factor'), 'x')} | occ {format_metric(report.get('clearance_occupied_percent'), '%', 1)}")
        lines.append(f"No-go: orange gap<{MIN_CLEARANCE_GAP_PX}px, purple center>{MAX_CENTER_OFFSET_RATIO:.2f}x")
    if report.get("reasons"):
        lines.append("Reason: " + "; ".join(report["reasons"][:2]))
    return lines


def draw_clearance_visualization(base_rgb: np.ndarray, report: dict[str, Any], include_footer: bool) -> Image.Image:
    overlay = base_rgb.astype(np.float32).copy()
    clearance_color = np.array([42, 157, 143], dtype=np.float32)
    dorn_color = np.array([69, 123, 157], dtype=np.float32)
    hit_color = np.array([230, 57, 70], dtype=np.float32)
    gap_color = np.array([255, 183, 3], dtype=np.float32)
    center_color = np.array([181, 23, 158], dtype=np.float32)
    clearance_mask = report.get("clearance_mask")
    dorn_mask = report.get("dorn_mask")
    outside_mask = report.get("outside_mask")
    if clearance_mask is not None and np.any(clearance_mask):
        overlay[clearance_mask] = 0.75 * overlay[clearance_mask] + 0.25 * clearance_color
    if dorn_mask is not None and np.any(dorn_mask):
        overlay[dorn_mask] = (1.0 - CLEARANCE_CHECK_ALPHA) * overlay[dorn_mask] + CLEARANCE_CHECK_ALPHA * dorn_color
    blend_mask_overlay(overlay, center_offset_forbidden_mask(report, overlay.shape), center_color, CENTER_FORBIDDEN_ALPHA)
    blend_mask_overlay(overlay, clearance_gap_forbidden_mask(report, overlay.shape), gap_color, GAP_FORBIDDEN_ALPHA)
    if not report.get("missing_classes") and outside_mask is not None and np.any(outside_mask):
        overlay[outside_mask] = 0.15 * overlay[outside_mask] + 0.85 * hit_color
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_image)
    draw_bbox(draw, report.get("clearance_bbox"), tuple(clearance_color.astype(int)), "StentClearance")
    draw_bbox(draw, report.get("dorn_bbox"), tuple(dorn_color.astype(int)), "Dorn")
    draw_clearance_threshold_guides(draw, report)
    if report.get("dorn_center") is not None and report.get("clearance_center") is not None:
        dorn_center = tuple(int(round(value)) for value in report["dorn_center"])
        clearance_center = tuple(int(round(value)) for value in report["clearance_center"])
        draw.line((dorn_center[0], dorn_center[1], clearance_center[0], clearance_center[1]), fill=(255, 255, 255), width=2)
    draw_center(draw, report.get("dorn_center"), tuple(dorn_color.astype(int)))
    draw_center(draw, report.get("clearance_center"), tuple(clearance_color.astype(int)))
    if not include_footer:
        return overlay_image
    lines = clearance_footer_lines(report)
    footer_height = max(90, 14 + 20 * len(lines))
    canvas = Image.new("RGB", (overlay_image.width, overlay_image.height + footer_height), (20, 20, 20))
    canvas.paste(overlay_image, (0, 0))
    canvas_draw = ImageDraw.Draw(canvas)
    status_color = (80, 220, 120) if report.get("pass_clearance") else (255, 90, 90)
    y = overlay_image.height + 8
    for index, line in enumerate(lines):
        canvas_draw.text((12, y), line, fill=status_color if index == 0 else (255, 255, 255))
        y += 20
    return canvas


def visualize_stent_clearance_fit(
    image_path: Path,
    report: dict[str, Any],
    output_dir: str | Path = DEFAULT_CLEARANCE_OUTPUT_DIR,
) -> Path:
    image = Image.open(resolve_path(image_path)).convert("RGB")
    canvas = draw_clearance_visualization(np.asarray(image, dtype=np.float32), report, include_footer=True)
    if canvas.width > 1400:
        scale = 1400 / canvas.width
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.BILINEAR)
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(image_path.stem)}_clearance_check.png"
    canvas.save(output_path)
    return output_path


def visualize_stent_clearance_fit_frame(frame_bgr: np.ndarray, report: dict[str, Any]) -> np.ndarray:
    cv2 = require_cv2()
    base_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    canvas = draw_clearance_visualization(base_rgb, report, include_footer=True)
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def put_fit_text(
    frame_bgr: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    base_scale: float = 0.64,
    thickness: int = 1,
) -> None:
    cv2 = require_cv2()
    x, y = origin
    max_width = max(40, frame_bgr.shape[1] - x - 12)
    scale = float(base_scale)
    while scale > 0.38:
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if text_width <= max_width:
            break
        scale -= 0.04
    cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def add_runtime_panel(
    frame_bgr: np.ndarray,
    report: dict[str, Any],
    fps: float,
    inference_ms: float,
    frame_count: int,
    total_latency_ms: float | None = None,
) -> np.ndarray:
    cv2 = require_cv2()
    shown = frame_bgr.copy()
    status = report.get("status", "n/a")
    pass_clearance = bool(report.get("pass_clearance"))
    status_color = (60, 220, 60) if pass_clearance else (40, 40, 255)
    if report.get("missing_classes"):
        detail = "Missing " + ", ".join(report["missing_classes"])
    else:
        detail = (
            f"out {format_metric(report.get('outside_percent'), '%', 1)} | "
            f"gap {format_metric(report.get('min_gap_px'), 'px', 0)}/{MIN_CLEARANCE_GAP_PX}px | "
            f"offset {format_metric(report.get('center_offset_px'), 'px', 1)}/{format_metric(report.get('center_limit_px'), 'px', 1)}"
        )
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 92), (18, 18, 18), -1)
    put_fit_text(shown, f"Stent clearance: {status}", (14, 29), status_color, base_scale=0.78, thickness=2)
    put_fit_text(shown, detail, (14, 57), (255, 255, 255), base_scale=0.56)
    if total_latency_ms is None:
        runtime_text = f"FPS {fps:.1f} | TRT {inference_ms:.1f} ms | Frame {frame_count}"
    else:
        runtime_options = [
            f"FPS {fps:.1f} | TRT {inference_ms:.1f} ms | Total {total_latency_ms:.1f} ms | Frame {frame_count}",
            f"FPS {fps:.1f} | TRT {inference_ms:.0f}ms | Total {total_latency_ms:.0f}ms | #{frame_count}",
            f"TRT {inference_ms:.0f}ms | Total {total_latency_ms:.0f}ms | #{frame_count}",
            f"Total {total_latency_ms:.0f}ms | #{frame_count}",
        ]
        max_width = max(40, shown.shape[1] - 26)
        runtime_text = runtime_options[-1]
        for option in runtime_options:
            text_width = cv2.getTextSize(option, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
            if text_width <= max_width:
                runtime_text = option
                break
    put_fit_text(shown, runtime_text, (14, 82), (210, 230, 255), base_scale=0.50)
    return shown
