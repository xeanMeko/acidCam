from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .batch import run_batch_mode
from .camera import run_camera_mode
from .clearance import run_clearance_smoke_tests
from .config import (
    DEFAULT_CLEARANCE_OUTPUT_DIR,
    DEFAULT_ENGINE,
    DEFAULT_INPUT,
    DEFAULT_MASK_OUTPUT_DIR,
    DEFAULT_OVERLAP_OUTPUT_DIR,
    DEFAULT_RAW_OUTPUT_DIR,
    DEFAULT_RUNTIME_CONFIG,
    WARMUP_RUNS,
)
from .preprocess import parse_crop_box


BOOL_KEYS = {
    "verbose",
    "gpio_dry_run",
    "no_gpio",
    "save_raw",
    "no_mask_overlays",
    "no_overlap_visuals",
    "no_clearance_visuals",
    "no_live_visualization",
    "self_test",
    "output_dry_run",
    "skip_missing_class_postprocess",
}

INT_KEYS = {
    "capture_width",
    "capture_height",
    "max_stale_frames",
    "print_every",
    "display_width",
    "display_every",
    "warmup_runs",
    "gpio_pin",
    "usb_relay_baudrate",
}

FLOAT_KEYS = {"stream_seconds", "capture_fps"}

CONTROL_KEYS = {"config", "write_default_config"}


def default_camera_source() -> str:
    return "/dev/video0" if Path("/dev/video0").exists() else "0"


def default_runtime_config() -> dict[str, Any]:
    return {
        "mode": "deploy",
        "engine": str(DEFAULT_ENGINE),
        "input": str(DEFAULT_INPUT),
        "camera": default_camera_source(),
        "capture_width": 3840,
        "capture_height": 2160,
        "max_stale_frames": 4,
        "capture_backend": "default",
        "capture_fourcc": "",
        "capture_fps": 0.0,
        "crop_box": "1500,900,2100,1480",
        "stream_seconds": 0.0,
        "print_every": 30,
        "display_width": 1200,
        "display_every": 1,
        "warmup_runs": WARMUP_RUNS,
        "verbose": False,

        # Output backend.
        # "gpio" = Jetson GPIO output
        # "usb"  = LCUS-1 / ARCELI USB relay
        # "none" = no physical output
        "output_mode": "gpio",
        "output_dry_run": False,

        # GPIO output.
        # Kept compatible with old config/CLI.
        "gpio_pin": None,
        "gpio_numbering": "BOARD",
        "gpio_dry_run": False,
        "no_gpio": False,

        # USB relay output.
        "usb_relay_port": "/dev/ttyUSB0",
        "usb_relay_baudrate": 9600,

        "save_raw": False,
        "raw_output_dir": str(DEFAULT_RAW_OUTPUT_DIR),
        "mask_output_dir": str(DEFAULT_MASK_OUTPUT_DIR),
        "overlap_output_dir": str(DEFAULT_OVERLAP_OUTPUT_DIR),
        "clearance_output_dir": str(DEFAULT_CLEARANCE_OUTPUT_DIR),
        "no_mask_overlays": False,
        "no_overlap_visuals": False,
        "no_clearance_visuals": False,
        "no_live_visualization": False,
        "skip_missing_class_postprocess": True,
        "self_test": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TensorRT Dorn/StentClearance live checker for Jetson Orin Nano."
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_RUNTIME_CONFIG),
        help="YAML runtime config file. CLI flags override values from this file.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write the built-in defaults to --config and exit.",
    )

    parser.add_argument("--mode", choices=("deploy", "debug", "visualizer", "batch"), default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--input", default=None, help="Image file or folder for batch mode.")
    parser.add_argument("--camera", default=None)

    parser.add_argument("--capture-width", type=int, default=None)
    parser.add_argument("--capture-height", type=int, default=None)
    parser.add_argument(
        "--capture-backend",
        choices=("default", "v4l2", "gstreamer"),
        default=None,
        help="OpenCV capture backend. On Jetson/Linux, v4l2 can reduce camera overhead for /dev/video* sources.",
    )
    parser.add_argument(
        "--capture-fourcc",
        default=None,
        help="Requested camera FOURCC, for example MJPG or YUYV. Empty leaves camera default.",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=None,
        help="Requested camera FPS. 0 leaves camera default.",
    )
    parser.add_argument(
        "--max-stale-frames",
        type=int,
        default=None,
        help="Maximum queued camera frames to discard before processing the next frame.",
    )
    parser.add_argument("--crop-box", default=None, help="x1,y1,x2,y2 or 'none'.")
    parser.add_argument(
        "--stream-seconds",
        type=float,
        default=None,
        help="0 runs until interrupted or q is pressed.",
    )
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument("--display-width", type=int, default=None, help="Reserved for notebook CLI parity.")
    parser.add_argument(
        "--display-every",
        type=int,
        default=None,
        help="Visualizer: redraw/imshow every N processed frames. Higher values reduce display overhead.",
    )
    parser.add_argument("--warmup-runs", type=int, default=None)

    parser.add_argument("--verbose", dest="verbose", action="store_true", default=None)
    parser.add_argument("--quiet", dest="verbose", action="store_false")

    # Output selection.
    parser.add_argument(
        "--output-mode",
        choices=("gpio", "usb", "none"),
        default=None,
        help="Physical output backend: gpio, usb, or none.",
    )
    parser.add_argument(
        "--output-dry-run",
        dest="output_dry_run",
        action="store_true",
        default=None,
        help="Print output actions without driving GPIO/USB relay.",
    )
    parser.add_argument(
        "--no-output-dry-run",
        dest="output_dry_run",
        action="store_false",
    )

    # GPIO output.
    parser.add_argument("--gpio-pin", type=int, default=None)
    parser.add_argument("--gpio-numbering", choices=("BOARD", "BCM"), default=None)
    parser.add_argument("--gpio-dry-run", dest="gpio_dry_run", action="store_true", default=None)
    parser.add_argument("--no-gpio-dry-run", dest="gpio_dry_run", action="store_false")
    parser.add_argument("--no-gpio", dest="no_gpio", action="store_true", default=None)
    parser.add_argument("--gpio", dest="no_gpio", action="store_false")

    # USB relay output.
    parser.add_argument(
        "--usb-relay-port",
        default=None,
        help="Serial port for LCUS-1 / ARCELI USB relay, e.g. /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--usb-relay-baudrate",
        type=int,
        default=None,
        help="Baudrate for USB relay. LCUS-1 default is 9600.",
    )

    parser.add_argument("--save-raw", dest="save_raw", action="store_true", default=None)
    parser.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    parser.add_argument("--raw-output-dir", default=None)
    parser.add_argument("--mask-output-dir", default=None)
    parser.add_argument("--overlap-output-dir", default=None)
    parser.add_argument("--clearance-output-dir", default=None)

    parser.add_argument("--no-mask-overlays", dest="no_mask_overlays", action="store_true", default=None)
    parser.add_argument("--mask-overlays", dest="no_mask_overlays", action="store_false")
    parser.add_argument("--no-overlap-visuals", dest="no_overlap_visuals", action="store_true", default=None)
    parser.add_argument("--overlap-visuals", dest="no_overlap_visuals", action="store_false")
    parser.add_argument("--no-clearance-visuals", dest="no_clearance_visuals", action="store_true", default=None)
    parser.add_argument("--clearance-visuals", dest="no_clearance_visuals", action="store_false")
    parser.add_argument(
        "--skip-missing-class-postprocess",
        dest="skip_missing_class_postprocess",
        action="store_true",
        default=None,
        help="Skip clearance mask post-processing when a required class is absent from high-confidence predictions.",
    )
    parser.add_argument(
        "--no-skip-missing-class-postprocess",
        dest="skip_missing_class_postprocess",
        action="store_false",
        help="Always run clearance mask post-processing even when a required class is absent.",
    )
    parser.add_argument(
        "--no-live-visualization",
        dest="no_live_visualization",
        action="store_true",
        default=None,
        help="Visualizer mode: show the raw live camera feed without clearance overlays or runtime panel.",
    )
    parser.add_argument(
        "--live-visualization",
        dest="no_live_visualization",
        action="store_false",
        help="Visualizer mode: show the live clearance visualization overlay.",
    )

    parser.add_argument(
        "--self-test",
        dest="self_test",
        action="store_true",
        default=None,
        help="Run synthetic clearance tests and exit.",
    )
    parser.add_argument("--no-self-test", dest="self_test", action="store_false")

    return parser


def coerce_config_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if key in BOOL_KEYS:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{key} must be true or false")

    if key in INT_KEYS:
        return int(value)

    if key in FLOAT_KEYS:
        return float(value)

    return value


def load_runtime_config(config_path: str | Path, parser: argparse.ArgumentParser) -> dict[str, Any]:
    path = Path(config_path).expanduser()

    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text()) or {}

    if not isinstance(data, dict):
        parser.error(f"Config file must contain a YAML mapping: {path}")

    allowed = set(default_runtime_config())
    normalized = {}
    unknown = []

    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")

        if key not in allowed:
            unknown.append(str(raw_key))
            continue

        try:
            normalized[key] = coerce_config_value(key, value)
        except (TypeError, ValueError) as exc:
            parser.error(f"Invalid config value for {raw_key}: {exc}")

    if unknown:
        parser.error(f"Unknown config key(s) in {path}: {', '.join(sorted(unknown))}")

    return normalized


def write_default_runtime_config(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(default_runtime_config(), sort_keys=False))

    return path


def merge_args_with_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.write_default_config:
        return args

    merged = default_runtime_config()
    merged.update(load_runtime_config(args.config, parser))

    for key, value in vars(args).items():
        if key in CONTROL_KEYS or value is None:
            continue
        merged[key] = value

    merged["config"] = args.config
    merged["write_default_config"] = args.write_default_config

    return argparse.Namespace(**merged)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.self_test:
        return

    camera_mode = args.mode in {"deploy", "debug", "visualizer"}

    # Backward compatibility:
    # Old --no-gpio now means no physical output.
    if args.no_gpio:
        args.output_mode = "none"

    # Backward compatibility:
    # Old --gpio-dry-run means GPIO output dry-run.
    if args.gpio_dry_run:
        args.output_mode = "gpio"
        args.output_dry_run = True

    if camera_mode:
        if args.output_mode == "gpio":
            if args.gpio_pin is None and not args.output_dry_run:
                parser.error("--gpio-pin is required for --output-mode gpio unless --output-dry-run is set")

        elif args.output_mode == "usb":
            if not args.usb_relay_port and not args.output_dry_run:
                parser.error("--usb-relay-port is required for --output-mode usb unless --output-dry-run is set")

        elif args.output_mode == "none":
            pass

        else:
            parser.error("--output-mode must be one of: gpio, usb, none")

    if args.max_stale_frames < 0:
        parser.error("--max-stale-frames must be >= 0")

    if args.print_every < 0:
        parser.error("--print-every must be >= 0")

    if args.display_every <= 0:
        parser.error("--display-every must be > 0")

    if args.capture_fps < 0:
        parser.error("--capture-fps must be >= 0")

    if args.capture_fourcc and len(str(args.capture_fourcc).strip()) != 4:
        parser.error("--capture-fourcc must be exactly four characters, for example MJPG or YUYV")

    if str(args.capture_backend).lower() not in {"default", "v4l2", "gstreamer"}:
        parser.error("--capture-backend must be one of: default, v4l2, gstreamer")

    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")

    if args.usb_relay_baudrate <= 0:
        parser.error("--usb-relay-baudrate must be > 0")

    parse_crop_box(args.crop_box)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    parsed_args = parser.parse_args(argv)

    if parsed_args.write_default_config:
        output_path = write_default_runtime_config(parsed_args.config)
        print(f"Wrote default runtime config: {output_path}")
        return 0

    args = merge_args_with_config(parsed_args, parser)
    validate_args(args, parser)

    if args.self_test:
        return run_clearance_smoke_tests()

    if args.mode == "batch":
        return run_batch_mode(args)

    frames = run_camera_mode(args)

    if args.mode != "deploy":
        print(f"Processed {frames} frame(s).")

    return 0