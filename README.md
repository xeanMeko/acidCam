# AcidCam

AcidCam is a Jetson-oriented RF-DETR segmentation runtime for checking whether the
`Dorn` mask fits inside the `StentClearance` mask. It can run against a live
camera, save batch visualizations for test images, and drive either Jetson GPIO or
an LCUS-1 / ARCELI USB relay as the pass/fail output.

## Repository Layout

```text
acidcam_runtime/              Runtime package used by run.py
scripts/export_rfdetr.py      RF-DETR ONNX export and TensorRT build helper
scripts/inference.py          ONNX Runtime benchmark helper
scripts/cpugpuTest.py         TensorRT benchmark / GPU telemetry helper
scripts/gpioTest.py           USB relay smoke test helper
convert_onnx_to_tensorRT.py   Small standalone ONNX-to-TensorRT converter
exports/rfdetr/               Exported ONNX / TensorRT model artifacts
test/                         Sample validation images and COCO annotations
notebooks/                    Exploratory notebooks
runtime_config.yaml           Default local runtime configuration
data.yaml                     Dataset class names and training metadata
```

Generated runtime outputs are written under `runs/`, which is intentionally
ignored by git.

## Environment

This project is intended for a Jetson runtime with CUDA, TensorRT, OpenCV, and
either Jetson GPIO or serial access for the chosen output device.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorRT engine files are tied to the TensorRT/CUDA/GPU stack that built them. If
an engine fails to deserialize on a target machine, rebuild it on that same
machine.

## Configuration

The runtime reads `runtime_config.yaml` by default. CLI flags override YAML
values.

Write a fresh config from built-in defaults:

```bash
python run.py --write-default-config
```

Useful config keys:

- `mode`: `visualizer`, `debug`, `deploy`, or `batch`
- `engine`: TensorRT engine path
- `input`: image or directory used by batch mode
- `camera`: camera index or device path
- `crop_box`: `x1,y1,x2,y2`, or `none`
- `output_mode`: `usb`, `gpio`, or `none`
- `output_dry_run`: print output actions without touching hardware
- `save_raw`: save raw TensorRT output tensors in batch mode

## Running

Batch mode over the sample images:

```bash
python run.py --mode batch --input test --output-mode none
```

Live visualizer:

```bash
python run.py --mode visualizer --config runtime_config.yaml
```

Headless deploy mode:

```bash
python run.py --mode deploy --config runtime_config.yaml
```

Run the synthetic clearance smoke tests:

```bash
python run.py --self-test
```

For dry hardware checks, keep the model/camera path active but avoid driving
physical outputs:

```bash
python run.py --mode visualizer --output-mode usb --output-dry-run
python run.py --mode visualizer --output-mode gpio --gpio-pin 15 --output-dry-run
```

## Model Export And TensorRT

The main export helper is `scripts/export_rfdetr.py`.

Export ONNX from a checkpoint:

```bash
python scripts/export_rfdetr.py \
  --checkpoint runs/rfdetr/rfdetr_seg_nano/checkpoint_best_ema.pth \
  --format onnx \
  --imgsz 312
```

Build a TensorRT engine with `trtexec`:

```bash
python scripts/export_rfdetr.py \
  --checkpoint runs/rfdetr/rfdetr_seg_nano/checkpoint_best_ema.pth \
  --format engine \
  --imgsz 312
```

For a quick conversion from an existing ONNX file:

```bash
python convert_onnx_to_tensorRT.py \
  --onnx exports/rfdetr/rfdetr-seg-nano_autocast_fp16.onnx \
  --engine exports/rfdetr/rfdetr-seg-nano_autocast_fp16.engine
```

## Outputs

Batch mode can save:

- raw TensorRT arrays: `runs/tensorrt_raw_outputs/`
- mask overlays: `runs/tensorrt_mask_overlays/`
- mask overlap visuals: `runs/tensorrt_overlap_visualizations/`
- clearance decision visuals: `runs/stent_clearance_checks/`

These directories are generated artifacts and are ignored by git.

## Notes

- Class names come from `data.yaml`; the runtime expects `Dorn` and
  `StentClearance`.
- `visualizer` opens an OpenCV window and exits on `q`.
- `deploy` is intended for continuous headless operation and prints only status
  changes.
- Rebuild TensorRT engines on the final target Jetson whenever CUDA, TensorRT, or
  JetPack changes.
