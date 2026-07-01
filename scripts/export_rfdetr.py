#!/usr/bin/env python3
"""Export RF-DETR checkpoints to ONNX and optionally TensorRT."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import shutil
import subprocess
import sys
import textwrap
from collections import deque
from pathlib import Path
from typing import NoReturn


def fail(message: str, exit_code: int = 1) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              Export ONNX for the trained RF-DETR nano segmentation checkpoint:
                python scripts/export_rfdetr.py \\
                  --checkpoint runs/rfdetr/rfdetr_seg_nano/checkpoint_best_ema.pth

              Export ONNX at a Jetson-friendly square resolution:
                python scripts/export_rfdetr.py \\
                  --checkpoint runs/rfdetr/rfdetr_seg_nano/checkpoint_best_ema.pth \\
                  --imgsz 312

              Build TensorRT on the Jetson that will run the model:
                python scripts/export_rfdetr.py \\
                  --checkpoint runs/rfdetr/rfdetr_seg_nano/checkpoint_best_ema.pth \\
                  --format engine \\
                  --imgsz 312
            """
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="RF-DETR .pth checkpoint to export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports") / "rfdetr",
        help="Directory for exported ONNX and TensorRT files.",
    )
    parser.add_argument(
        "--format",
        choices=("onnx", "engine", "both"),
        default="onnx",
        help="Export ONNX only, TensorRT engine only, or both. Engine export always creates/uses ONNX first.",
    )

    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--imgsz",
        type=positive_int,
        default=None,
        help="Square export size. If omitted, RF-DETR uses the checkpoint/model resolution.",
    )
    shape_group.add_argument(
        "--shape",
        type=positive_int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Explicit export shape. Both dimensions must be valid for the selected RF-DETR variant.",
    )

    parser.add_argument("--batch-size", type=positive_int, default=1, help="Static export batch size.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Export ONNX with a dynamic batch dimension.")
    parser.add_argument("--opset", type=positive_int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used while tracing ONNX. CPU is portable; use cuda/cuda:0 only where CUDA is available.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose RF-DETR/ONNX export output.")
    parser.add_argument(
        "--onnx-fp16",
        action="store_true",
        help="Convert the exported ONNX to mixed-precision FP16 while keeping inputs/outputs FP32.",
    )

    parser.add_argument("--trtexec", default="trtexec", help="Path or command name for NVIDIA trtexec.")
    parser.add_argument(
        "--workspace-mib",
        type=positive_int,
        default=2048,
        help="TensorRT workspace size in MiB for --memPoolSize=workspace:<MiB>.",
    )
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True, help="Enable TensorRT FP16 mode.")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable TensorRT FP16 mode.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the TensorRT command without running trtexec. ONNX export still runs if needed.",
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        fail(f"{description} not found: {resolved}")
    if not resolved.is_file():
        fail(f"{description} must be a file: {resolved}")
    return resolved


def resolve_shape(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.shape is not None:
        return tuple(args.shape)
    if args.imgsz is not None:
        return (args.imgsz, args.imgsz)
    return None


def require_rfdetr():
    try:
        from rfdetr import RFDETR
    except Exception as exc:
        fail(
            "RF-DETR could not be imported. Activate an environment where RF-DETR works "
            "(for example `conda activate rfdetr` or a fixed `yolo26` env), then install "
            "the project requirements. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
    return RFDETR


def require_onnx_export_deps() -> None:
    if not has_module("onnx"):
        fail(
            "ONNX export dependencies are missing. Install the RF-DETR ONNX extra with "
            "`python -m pip install 'rfdetr[onnx]'`, or rerun "
            "`python -m pip install -r requirements.txt` after updating the environment."
        )


def set_export_device(model, device_name: str) -> None:
    try:
        import torch
    except Exception as exc:
        fail(f"PyTorch could not be imported from this environment. Original error: {type(exc).__name__}: {exc}")

    try:
        device = torch.device(device_name)
    except Exception as exc:
        fail(f"Invalid export device {device_name!r}. Original error: {type(exc).__name__}: {exc}")

    model_context = getattr(model, "model", None)
    torch_model = getattr(model_context, "model", None)
    if model_context is None or torch_model is None:
        fail("Could not find the underlying RF-DETR model context needed for export.")

    try:
        model_context.device = device
        model_context.model = torch_model.to(device)
        if getattr(model_context, "inference_model", None) is not None:
            model_context.inference_model = model_context.inference_model.to(device)
    except RuntimeError as exc:
        fail(
            f"Could not move RF-DETR to export device {device_name!r}. "
            "Use `--device cpu` on machines without CUDA. Original error: "
            f"{exc}"
        )


def resolve_trtexec(command: str) -> str | None:
    command_path = Path(command).expanduser()
    if command_path.parent != Path("."):
        return str(command_path) if command_path.exists() else None
    return shutil.which(command)


def require_trtexec(args: argparse.Namespace) -> str:
    executable = resolve_trtexec(args.trtexec)
    if executable is not None:
        return executable

    fail(
        "TensorRT engine export was requested, but `trtexec` was not found. "
        "Build the engine on the Orin Nano where TensorRT is installed, for example:\n"
        "  trtexec --onnx=exports/rfdetr/rfdetr-seg-nano.onnx "
        "--saveEngine=exports/rfdetr/rfdetr-seg-nano.engine "
        "--fp16 --memPoolSize=workspace:2048\n"
        "or pass `--trtexec /path/to/trtexec`."
    )


def run_trtexec(command: list[str]) -> tuple[int, str]:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        fail(f"Could not start trtexec. Original error: {type(exc).__name__}: {exc}")

    log_tail: deque[str] = deque(maxlen=240)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_tail.append(line)

    return process.wait(), "".join(log_tail)


def diagnose_trtexec_failure(log_tail: str) -> str | None:
    lowered = log_tail.lower()
    if "unsupported display driver / cuda driver combination" in lowered:
        return textwrap.dedent(
            """\
            trtexec started, but CUDA could not initialize. On Jetson this usually
            means the CUDA runtime visible to this shell does not match the installed
            JetPack/L4T driver stack.

            Quick checks on the Jetson:
              cat /etc/nv_tegra_release
              dpkg -l | grep -E 'nvidia-l4t-core|cuda|tensorrt|libnvinfer'
              echo "$LD_LIBRARY_PATH"
              ldd /usr/src/tensorrt/bin/trtexec | grep -E 'cuda|nvinfer'

            If Conda or pip CUDA libraries appear ahead of the JetPack libraries,
            run trtexec from a clean shell or remove those paths from LD_LIBRARY_PATH.
            If the JetPack packages themselves are mixed, reinstall the matching
            JetPack/CUDA/TensorRT set for this device.
            """
        ).strip()

    if "no importer registered for op: layernormalization" in lowered or "plugin: layernormalization" in lowered:
        return textwrap.dedent(
            """\
            CUDA initialized, but TensorRT could not parse the ONNX graph because
            it contains a LayerNormalization op without a matching TensorRT importer
            or plugin on this Jetson.

            First try exporting ONNX with an older opset so layer norm is more
            likely to be decomposed into primitive ONNX ops:
              python scripts/export_rfdetr.py --checkpoint <checkpoint.pth> --opset 16 --imgsz 312

            Then build the TensorRT engine from that new ONNX file in a clean
            shell outside Conda. If TensorRT still reports LayerNormalization,
            use a newer JetPack/TensorRT stack or an ONNX graph/plugin workflow
            that replaces or implements LayerNormalization for TensorRT.
            """
        ).strip()

    return None


def export_onnx(args: argparse.Namespace, checkpoint: Path) -> Path:
    RFDETR = require_rfdetr()
    require_onnx_export_deps()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "output_dir": str(output_dir),
        "format": "onnx",
        "opset_version": args.opset,
        "batch_size": args.batch_size,
        "dynamic_batch": args.dynamic_batch,
        "verbose": args.verbose,
    }
    shape = resolve_shape(args)
    if shape is not None:
        export_kwargs["shape"] = shape

    print(f"Loading RF-DETR checkpoint: {checkpoint}")
    model = RFDETR.from_checkpoint(str(checkpoint))
    set_export_device(model, args.device)

    print(f"Exporting ONNX to: {output_dir}")
    try:
        onnx_path = model.export(**export_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        fail(
            "RF-DETR ONNX export dependencies are missing. Install them with "
            "`python -m pip install 'rfdetr[onnx]'`. Original error: "
            f"{type(exc).__name__}: {exc}"
        )
    except RuntimeError as exc:
        if "aten::_upsample_bicubic2d_aa" in str(exc):
            fail(
                "ONNX export failed while resizing RF-DETR positional embeddings. "
                "For this checkpoint, omit `--imgsz`/`--shape` or use its native "
                "export size; export a checkpoint trained at the target resolution "
                f"for other sizes. Original error: {exc}"
            )
        if "onnx" in str(exc).lower():
            fail(
                "ONNX export failed. Make sure `rfdetr[onnx]` is installed in this "
                f"environment. Original error: {exc}"
            )
        raise
    except ValueError as exc:
        fail(f"Invalid RF-DETR export options: {exc}")

    return Path(onnx_path).expanduser().resolve()


def convert_onnx_to_fp16(onnx_path: Path) -> Path:
    fp16_path = onnx_path.with_name(f"{onnx_path.stem}-fp16{onnx_path.suffix}")

    if has_module("modelopt.onnx.autocast"):
        try:
            import modelopt.onnx.autocast as autocast
            import onnx
        except (ImportError, ModuleNotFoundError) as exc:
            fail(
                "ONNX FP16 conversion requested, but NVIDIA ModelOpt ONNX dependencies "
                "could not be imported. Install them with "
                "`python -m pip install 'nvidia-modelopt[onnx]'`. Original error: "
                f"{type(exc).__name__}: {exc}"
            )

        print(f"Converting ONNX to mixed-precision FP16 with NVIDIA ModelOpt: {fp16_path}")
        converted_model = autocast.convert_to_mixed_precision(
            onnx_path=str(onnx_path),
            low_precision_type="fp16",
            keep_io_types=True,
        )
        onnx.save(converted_model, str(fp16_path))
        return fp16_path.resolve()

    if has_module("onnxconverter_common.float16"):
        try:
            import onnx
            from onnxconverter_common import float16
        except (ImportError, ModuleNotFoundError) as exc:
            fail(
                "ONNX FP16 conversion requested, but onnxconverter-common could not "
                "be imported. Install it with "
                "`python -m pip install onnx onnxconverter-common`. Original error: "
                f"{type(exc).__name__}: {exc}"
            )

        print(f"Converting ONNX to FP16 with onnxconverter-common: {fp16_path}")
        model = onnx.load(str(onnx_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(fp16_path))
        return fp16_path.resolve()

    fail(
        "ONNX FP16 conversion was requested, but no supported converter is installed. "
        "Install the NVIDIA-recommended converter with "
        "`python -m pip install 'nvidia-modelopt[onnx]'`, or install the fallback "
        "converter with `python -m pip install onnx onnxconverter-common`."
    )


def build_tensorrt_engine(args: argparse.Namespace, onnx_path: Path, trtexec: str) -> Path:
    engine_path = onnx_path.with_suffix(".engine")
    command = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]
    if args.fp16 and args.onnx_fp16:
        print("Skipping trtexec --fp16 because the ONNX model already carries FP16 types.")
    elif args.fp16:
        command.append("--fp16")

    print("TensorRT command:")
    print(shlex.join(str(part) for part in command))
    if args.dry_run:
        print("Dry run requested; not running trtexec.")
        return engine_path

    return_code, log_tail = run_trtexec(command)
    if return_code != 0:
        diagnostic = diagnose_trtexec_failure(log_tail)
        if diagnostic is not None:
            fail(f"trtexec failed with exit code {return_code}\n\n{diagnostic}")
        fail(f"trtexec failed with exit code {return_code}")
    return engine_path


def main() -> None:
    args = parse_args()
    checkpoint = resolve_existing_path(args.checkpoint, "RF-DETR checkpoint")

    trtexec = None
    if args.format in {"engine", "both"}:
        trtexec = require_trtexec(args)

    onnx_path = export_onnx(args, checkpoint)
    print(f"ONNX export complete: {onnx_path}")

    if args.onnx_fp16:
        onnx_path = convert_onnx_to_fp16(onnx_path)
        print(f"FP16 ONNX path: {onnx_path}")

    if args.format in {"engine", "both"}:
        assert trtexec is not None
        engine_path = build_tensorrt_engine(args, onnx_path, trtexec)
        print(f"TensorRT engine path: {engine_path}")


if __name__ == "__main__":
    main()
