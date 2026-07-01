from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import INPUT_SHAPE, resolve_path

_TRT = None
_CUDART = None


def require_trt_cuda():
    global _TRT, _CUDART
    if _TRT is None:
        import tensorrt as trt

        _TRT = trt
    if _CUDART is None:
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            from cuda import cudart

        _CUDART = cudart
    return _TRT, _CUDART


def check_cuda(result):
    _, cudart = require_trt_cuda()
    if isinstance(result, tuple):
        err, values = result[0], result[1:]
    else:
        err, values = result, ()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return values[0] if len(values) == 1 else values


def volume(shape: tuple[int, ...]) -> int:
    return int(np.prod([int(dim) for dim in shape]))


def engine_rebuild_hint(engine_path: Path) -> str:
    return (
        f"Failed to deserialize TensorRT engine: {engine_path}\n"
        "TensorRT .engine files are tied to the TensorRT/CUDA/GPU runtime that built them. "
        "Rebuild the engine on this machine/runtime, for example:\n"
        "  trtexec --onnx=exports/rfdetr/rfdetr-seg-nano.onnx "
        "--saveEngine=exports/rfdetr/rfdetr-seg-nano.engine --fp16 --memPoolSize=workspace:2048"
    )


def load_engine(engine_path: str | Path):
    trt, _ = require_trt_cuda()
    path = resolve_path(engine_path)
    if not path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {path}")
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    try:
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
    except Exception as exc:
        raise RuntimeError(engine_rebuild_hint(path)) from exc
    if engine is None:
        raise RuntimeError(engine_rebuild_hint(path))
    return engine


def get_io_names(engine) -> tuple[list[str], list[str]]:
    trt, _ = require_trt_cuda()
    inputs, outputs = [], []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def allocate_buffers(engine, context, input_shape: tuple[int, int, int, int]):
    trt, cudart = require_trt_cuda()
    input_names, output_names = get_io_names(engine)
    if len(input_names) != 1:
        raise RuntimeError(f"Expected 1 input, found {len(input_names)}: {input_names}")
    input_name = input_names[0]
    bindings: dict[str, dict[str, Any]] = {}
    stream = None
    try:
        if context.set_input_shape(input_name, input_shape) is False:
            raise RuntimeError(f"TensorRT rejected input shape {input_shape} for {input_name}")
        stream = check_cuda(cudart.cudaStreamCreate())
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
        input_nbytes = volume(input_shape) * np.dtype(input_dtype).itemsize
        bindings[input_name] = {
            "host": None,
            "device": check_cuda(cudart.cudaMalloc(input_nbytes)),
            "shape": tuple(input_shape),
            "dtype": input_dtype,
            "nbytes": input_nbytes,
            "is_input": True,
        }
        for output_name in output_names:
            output_shape = tuple(context.get_tensor_shape(output_name))
            if any(dim < 0 for dim in output_shape):
                raise RuntimeError(f"Output shape for {output_name} is still dynamic: {output_shape}")
            output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
            output_size = volume(output_shape)
            output_nbytes = output_size * np.dtype(output_dtype).itemsize
            bindings[output_name] = {
                "host": np.empty(output_size, dtype=output_dtype),
                "device": check_cuda(cudart.cudaMalloc(output_nbytes)),
                "shape": output_shape,
                "dtype": output_dtype,
                "nbytes": output_nbytes,
                "is_input": False,
            }
        for name, buffer in bindings.items():
            if context.set_tensor_address(name, int(buffer["device"])) is False:
                raise RuntimeError(f"TensorRT rejected device address for tensor {name}")
        return input_name, output_names, bindings, stream
    except Exception:
        for buffer in bindings.values():
            if buffer.get("device") is not None:
                check_cuda(cudart.cudaFree(buffer["device"]))
        if stream is not None:
            check_cuda(cudart.cudaStreamDestroy(stream))
        raise


def infer(context, input_name: str, output_names: list[str], bindings: dict, stream, image: np.ndarray) -> list[np.ndarray]:
    _, cudart = require_trt_cuda()
    input_buffer = bindings[input_name]
    if image.dtype != input_buffer["dtype"]:
        image = image.astype(input_buffer["dtype"], copy=False)
    image = np.ascontiguousarray(image)
    if tuple(image.shape) != tuple(input_buffer["shape"]):
        raise ValueError(f"Input shape mismatch. Expected {input_buffer['shape']}, got {image.shape}")
    check_cuda(
        cudart.cudaMemcpyAsync(
            input_buffer["device"],
            image.ctypes.data,
            input_buffer["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
    )
    if not context.execute_async_v3(stream_handle=stream):
        raise RuntimeError("TensorRT execution failed")
    for output_name in output_names:
        output_buffer = bindings[output_name]
        check_cuda(
            cudart.cudaMemcpyAsync(
                output_buffer["host"].ctypes.data,
                output_buffer["device"],
                output_buffer["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )
    check_cuda(cudart.cudaStreamSynchronize(stream))
    return [bindings[name]["host"].reshape(bindings[name]["shape"]).copy() for name in output_names]


def free_buffers(bindings: dict | None, stream) -> None:
    if bindings is None:
        return
    _, cudart = require_trt_cuda()
    for buffer in bindings.values():
        if buffer.get("device") is not None:
            check_cuda(cudart.cudaFree(buffer["device"]))
            buffer["device"] = None
    if stream is not None:
        check_cuda(cudart.cudaStreamDestroy(stream))


class TensorRTSession:
    def __init__(self, engine_path: str | Path):
        self.engine_path = resolve_path(engine_path)
        self.engine = None
        self.context = None
        self.input_name = ""
        self.output_names: list[str] = []
        self.bindings = None
        self.stream = None

    def __enter__(self) -> "TensorRTSession":
        self.engine = load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self.input_name, self.output_names, self.bindings, self.stream = allocate_buffers(
            self.engine,
            self.context,
            INPUT_SHAPE,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.bindings is not None:
            free_buffers(self.bindings, self.stream)
            self.bindings = None
            self.stream = None

    def run(self, image: np.ndarray) -> dict[str, np.ndarray]:
        outputs = infer(self.context, self.input_name, self.output_names, self.bindings, self.stream, image)
        return dict(zip(self.output_names, outputs))

    def warmup(self, sample: np.ndarray, runs: int) -> None:
        for _ in range(max(0, int(runs))):
            self.run(sample)

    def print_metadata(self) -> None:
        input_buffer = self.bindings[self.input_name]
        print(f"Input: {self.input_name}, shape={input_buffer['shape']}, dtype={input_buffer['dtype']}")
        print("Outputs:")
        for output_name in self.output_names:
            output_buffer = self.bindings[output_name]
            print(f"  {output_name}: shape={output_buffer['shape']}, dtype={output_buffer['dtype']}")
