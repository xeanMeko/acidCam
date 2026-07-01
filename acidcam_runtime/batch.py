from __future__ import annotations

import argparse
import timeit

import numpy as np

from .clearance import analyze_stent_clearance_fit
from .config import CLASS_NAMES, PROJECT_ROOT, collect_images, display_path, resolve_path
from .postprocess import save_mask_overlay, save_raw_outputs, tensor_stats
from .preprocess import preprocess_path
from .trt_runtime import TensorRTSession
from .visualization import visualize_mask_overlap, visualize_stent_clearance_fit


def _elapsed_ms(start: float) -> float:
    return (timeit.default_timer() - start) * 1000.0


def run_batch_mode(args: argparse.Namespace) -> int:
    images = collect_images(args.input)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Found {len(images)} image(s) under {display_path(resolve_path(args.input))}")
    print(f"Loading TensorRT engine: {resolve_path(args.engine)}")
    per_image_times = []
    first_output_map = None
    saved_raw = saved_masks = saved_overlaps = saved_clearance = 0
    with TensorRTSession(args.engine) as runner:
        runner.print_metadata()
        sample = preprocess_path(images[0])
        runner.warmup(sample, args.warmup_runs)
        if args.warmup_runs:
            print(f"Warmup runs: {args.warmup_runs}")

        for image_path in images:
            preprocess_start = timeit.default_timer()
            image = preprocess_path(image_path)
            preprocessing_ms = _elapsed_ms(preprocess_start)

            inference_start = timeit.default_timer()
            output_map = runner.run(image)
            inference_ms = _elapsed_ms(inference_start)

            if first_output_map is None:
                first_output_map = output_map

            postprocess_ms = 0.0
            visualization_ms = 0.0

            print(f"\n{display_path(image_path)}: inference {inference_ms:.3f} ms")

            if args.save_raw:
                postprocess_start = timeit.default_timer()
                raw_path = save_raw_outputs(image_path, output_map, args.raw_output_dir)
                postprocess_ms += _elapsed_ms(postprocess_start)
                saved_raw += 1
                print(f"  raw outputs: {display_path(raw_path)}")

            if not args.no_mask_overlays:
                visualization_start = timeit.default_timer()
                overlay_path, mask_count = save_mask_overlay(image_path, output_map, args.mask_output_dir)
                visualization_ms += _elapsed_ms(visualization_start)
                saved_masks += 1
                print(f"  mask overlay: {display_path(overlay_path)} ({mask_count} mask(s))")

            if not args.no_overlap_visuals:
                visualization_start = timeit.default_timer()
                overlap_path, overlap = visualize_mask_overlap(output_map, image_path, args.overlap_output_dir)
                visualization_ms += _elapsed_ms(visualization_start)
                saved_overlaps += 1
                print(f"  overlap: {overlap['overall_overlap_percent']:.2f}% of union; {display_path(overlap_path)}")

            postprocess_start = timeit.default_timer()
            report = analyze_stent_clearance_fit(
                output_map,
                image_path,
                skip_missing_class_postprocess=bool(getattr(args, "skip_missing_class_postprocess", True)),
            )
            postprocess_ms += _elapsed_ms(postprocess_start)

            print(f"  clearance: {report['status']}")
            for reason in report.get("reasons", []):
                print(f"    - {reason}")

            if not args.no_clearance_visuals:
                visualization_start = timeit.default_timer()
                clearance_path = visualize_stent_clearance_fit(image_path, report, args.clearance_output_dir)
                visualization_ms += _elapsed_ms(visualization_start)
                saved_clearance += 1
                print(f"  clearance visual: {display_path(clearance_path)}")

            total_ms = preprocessing_ms + inference_ms + postprocess_ms + visualization_ms
            per_image_times.append(
                {
                    "image_path": image_path,
                    "preprocessing_ms": preprocessing_ms,
                    "inference_ms": inference_ms,
                    "postprocess_ms": postprocess_ms,
                    "visualization_ms": visualization_ms,
                    "total_ms": total_ms,
                }
            )

            print(
                "  latency: "
                f"total {total_ms:.3f} ms; "
                f"preprocessing {preprocessing_ms:.3f} ms; "
                f"inference {inference_ms:.3f} ms; "
                f"post-processing {postprocess_ms:.3f} ms; "
                f"visualization {visualization_ms:.3f} ms"
            )

    if first_output_map is not None:
        print("\nFirst image output stats:")
        for name, array in first_output_map.items():
            print(f"  {tensor_stats(name, array)}")

    print("\nPer-image inference times:")
    for record in per_image_times:
        print(f"  {display_path(record['image_path'])}: {record['inference_ms']:.3f} ms")

    total_times = np.array([record["total_ms"] for record in per_image_times], dtype=np.float64)
    preprocessing_times = np.array([record["preprocessing_ms"] for record in per_image_times], dtype=np.float64)
    inference_times = np.array([record["inference_ms"] for record in per_image_times], dtype=np.float64)
    postprocess_times = np.array([record["postprocess_ms"] for record in per_image_times], dtype=np.float64)
    visualization_times = np.array([record["visualization_ms"] for record in per_image_times], dtype=np.float64)

    avg_total = float(total_times.mean())
    avg_preprocessing = float(preprocessing_times.mean())
    avg_inference = float(inference_times.mean())
    avg_postprocess = float(postprocess_times.mean())
    avg_visualization = float(visualization_times.mean())
    fps = 1000.0 / avg_inference if avg_inference > 0 else float("inf")

    print("\n==============================")
    print(f"Images processed: {len(images)}")
    print(f"Average total latency: {avg_total:.3f} ms")
    print(f"Average preprocessing latency: {avg_preprocessing:.3f} ms")
    print(f"Average inference latency: {avg_inference:.3f} ms")
    print(f"Average post-processing latency: {avg_postprocess:.3f} ms")
    print(f"Average visualization latency: {avg_visualization:.3f} ms")
    print(f"TensorRT FPS: {fps:.2f}")

    if saved_raw:
        print(f"Saved raw outputs to: {resolve_path(args.raw_output_dir)}")
    if saved_masks:
        print(f"Saved mask overlays to: {resolve_path(args.mask_output_dir)}")
    if saved_overlaps:
        print(f"Saved overlap visualizations to: {resolve_path(args.overlap_output_dir)}")
    if saved_clearance:
        print(f"Saved clearance visualizations to: {resolve_path(args.clearance_output_dir)}")
    print("==============================")
    return 0