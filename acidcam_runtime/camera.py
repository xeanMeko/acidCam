from __future__ import annotations

import argparse
import time
from typing import Any

from .clearance import analyze_stent_clearance_fit, format_metric, frame_output_map_for_clearance
from .config import CLASS_NAMES, MIN_CLEARANCE_GAP_PX, PROJECT_ROOT, resolve_path
from .output_control import GpioOutput, OutputBase, UsbRelayOutput
from .preprocess import (
    configure_capture,
    crop_camera_frame,
    parse_camera_source,
    parse_crop_box,
    preprocess_bgr_frame,
    require_cv2,
)
from .trt_runtime import TensorRTSession
from .visualization import add_runtime_panel, visualize_stent_clearance_fit_frame


class DisabledOutput(OutputBase):
    """
    No physical output. Keeps the same interface as GPIO/USB relay.
    """

    def __init__(self) -> None:
        self.state: bool | None = False

    def write(self, passed: bool) -> None:
        self.state = bool(passed)

    def cleanup(self) -> None:
        self.state = False


def create_output(args: argparse.Namespace) -> OutputBase:
    mode = str(getattr(args, "output_mode", "gpio")).lower()

    # Backward compatibility with old config/CLI.
    if getattr(args, "no_gpio", False):
        mode = "none"

    output_dry_run = bool(getattr(args, "output_dry_run", False))

    # Backward compatibility:
    # old gpio_dry_run also enables output dry-run for GPIO.
    if getattr(args, "gpio_dry_run", False):
        mode = "gpio"
        output_dry_run = True

    if mode == "gpio":
        return GpioOutput(
            pin=args.gpio_pin,
            numbering=args.gpio_numbering,
            dry_run=output_dry_run,
            disabled=False,
        )

    if mode == "usb":
        return UsbRelayOutput(
            port=args.usb_relay_port,
            baudrate=args.usb_relay_baudrate,
            dry_run=output_dry_run,
            disabled=False,
        )

    if mode == "none":
        return DisabledOutput()

    raise ValueError(f"Unknown output_mode: {mode!r}")


def output_state_label(output: OutputBase) -> str:
    if output.state is None:
        return "UNKNOWN"
    return "ON" if output.state else "OFF"


def capture_backend_id(cv2, backend: str | None) -> int | None:
    name = str(backend or "default").strip().lower()
    if not name or name == "default":
        return None
    if name == "v4l2":
        return cv2.CAP_V4L2
    if name == "gstreamer":
        return cv2.CAP_GSTREAMER
    raise ValueError(f"Unknown capture_backend: {backend!r}")


def print_live_report(
    frame_count: int,
    report: dict[str, Any],
    fps: float,
    inference_ms: float,
    output: OutputBase,
) -> None:
    if report.get("missing_classes"):
        detail = "missing=" + ",".join(report["missing_classes"])
    else:
        detail = (
            f"outside={format_metric(report.get('outside_percent'), '%', 1)} "
            f"gap={format_metric(report.get('min_gap_px'), 'px', 0)}/{MIN_CLEARANCE_GAP_PX}px "
            f"offset={format_metric(report.get('center_offset_px'), 'px', 1)}/"
            f"{format_metric(report.get('center_limit_px'), 'px', 1)} "
            f"area={format_metric(report.get('area_factor'), 'x')}"
        )

    print(
        f"frame={frame_count} "
        f"status={report.get('status')} "
        f"output={output_state_label(output)} "
        f"fps={fps:.1f} "
        f"trt={inference_ms:.1f}ms "
        f"{detail}"
    )

    for reason in report.get("reasons", []):
        print(f"  - {reason}")


def capture_fps_hint(cv2, cap) -> float:
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0

    return fps if 1.0 <= fps <= 240.0 else 30.0


def stale_frame_grabs(last_capture_at: float | None, capture_fps: float, max_stale_frames: int) -> int:
    if last_capture_at is None or capture_fps <= 0.0 or max_stale_frames <= 0:
        return 0

    elapsed = max(0.0, time.perf_counter() - last_capture_at)
    frames_elapsed = int(elapsed * capture_fps)

    return min(max_stale_frames, max(0, frames_elapsed - 1))


def read_latest_frame(
    cap,
    last_capture_at: float | None,
    capture_fps: float,
    max_stale_frames: int,
):
    for _ in range(stale_frame_grabs(last_capture_at, capture_fps, max_stale_frames)):
        if not cap.grab():
            return False, None, last_capture_at

    ok, frame = cap.read()

    return ok, frame, time.perf_counter() if ok else last_capture_at


def run_camera_mode(args: argparse.Namespace) -> int:
    cv2 = require_cv2()
    crop_box = parse_crop_box(args.crop_box)

    show_window = args.mode == "visualizer"
    print_reports = args.mode in {"debug", "visualizer"}
    live_visualization = not bool(getattr(args, "no_live_visualization", False))
    display_every = max(1, int(getattr(args, "display_every", 1)))

    output = create_output(args)

    cap = None
    last_status = None
    last_capture_at: float | None = None
    frame_count = 0
    start_time = time.perf_counter()
    window_name = (
        "Dorn/StentClearance clearance visualizer (press q to stop)"
        if live_visualization
        else "Dorn/StentClearance live feed (press q to stop)"
    )

    try:
        output.fail_safe_low()

        with TensorRTSession(args.engine) as runner:
            if args.verbose or args.mode != "deploy":
                print(f"Project root: {PROJECT_ROOT}")
                print(f"Classes: {CLASS_NAMES}")
                print(f"Engine: {resolve_path(args.engine)}")
                print(f"Output mode: {getattr(args, 'output_mode', 'gpio')}")

                if getattr(args, "output_mode", "gpio") == "usb":
                    print(f"USB relay: {args.usb_relay_port} @ {args.usb_relay_baudrate} baud")
                elif getattr(args, "output_mode", "gpio") == "gpio":
                    print(f"GPIO: {args.gpio_numbering} pin {args.gpio_pin}")

                runner.print_metadata()

            camera_source = parse_camera_source(args.camera)
            backend_id = capture_backend_id(cv2, getattr(args, "capture_backend", "default"))
            cap = cv2.VideoCapture(camera_source, backend_id) if backend_id is not None else cv2.VideoCapture(camera_source)
            configure_capture(
                cap,
                args.capture_width,
                args.capture_height,
                getattr(args, "capture_fps", 0.0),
                getattr(args, "capture_fourcc", ""),
            )

            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera source: {args.camera}")

            capture_fps = capture_fps_hint(cv2, cap)
            max_stale_frames = max(0, int(args.max_stale_frames))

            if args.mode != "deploy":
                print(
                    f"Mode: {args.mode}; "
                    f"camera: {args.camera}; "
                    f"capture: {args.capture_width}x{args.capture_height}; "
                    f"backend: {getattr(args, 'capture_backend', 'default')}; "
                    f"fourcc: {getattr(args, 'capture_fourcc', '') or 'default'}; "
                    f"fps request: {float(getattr(args, 'capture_fps', 0.0) or 0.0):.1f}; "
                    f"crop: {crop_box}; "
                    f"display every: {display_every}; "
                    f"live visualization: {'on' if live_visualization else 'off'}; "
                    f"stale frame drain: {max_stale_frames} @ {capture_fps:.1f}fps"
                )

            while True:
                if args.stream_seconds and time.perf_counter() - start_time >= float(args.stream_seconds):
                    print("Stream time limit reached.")
                    break

                ok, frame, last_capture_at = read_latest_frame(
                    cap,
                    last_capture_at,
                    capture_fps,
                    max_stale_frames,
                )

                if not ok or frame is None:
                    output.fail_safe_low()

                    if args.mode != "deploy":
                        print("No frame received from camera. Continuing...")

                    time.sleep(0.02)
                    continue

                frame_latency_start = time.perf_counter()

                frame = crop_camera_frame(frame, crop_box)
                image = preprocess_bgr_frame(frame)

                inference_start = time.perf_counter()
                output_map = runner.run(image)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0

                report = analyze_stent_clearance_fit(
                    frame_output_map_for_clearance(output_map, frame.shape),
                    skip_missing_class_postprocess=bool(getattr(args, "skip_missing_class_postprocess", True)),
                )

                passed = bool(report.get("pass_clearance"))
                output.write(passed)

                frame_count += 1

                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0.0

                status = str(report.get("status", "n/a"))

                should_print = False

                if print_reports and args.print_every > 0 and frame_count % int(args.print_every) == 0:
                    should_print = True

                if args.mode == "deploy" and status != last_status:
                    should_print = True

                total_latency_ms = (time.perf_counter() - frame_latency_start) * 1000.0

                if should_print:
                    print_live_report(frame_count, report, fps, inference_ms, output)

                last_status = status

                if show_window:
                    if frame_count % display_every == 0:
                        if live_visualization:
                            shown = visualize_stent_clearance_fit_frame(frame, report)
                            total_latency_ms = (time.perf_counter() - frame_latency_start) * 1000.0
                            shown = add_runtime_panel(
                                shown,
                                report,
                                fps,
                                inference_ms,
                                frame_count,
                                total_latency_ms,
                            )
                        else:
                            shown = frame

                        cv2.imshow(window_name, shown)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Stream stopped by user.")
                        break

    except KeyboardInterrupt:
        print("Stream stopped by keyboard interrupt.")

    except Exception:
        output.fail_safe_low()
        raise

    finally:
        if cap is not None:
            cap.release()

        if show_window:
            cv2.destroyAllWindows()

        output.cleanup()

    return frame_count