import os
import timeit
import numpy as np
from PIL import Image
import tqdm
import tensorrt as trt
from jtop import jtop

# CUDA Python import: supports both newer and older package layouts
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


# ----------------------------
# Model + data
# ----------------------------
engine_path = "exports/rfdetr/rfdetr-seg-nano.engine"
image_folder = "test/"

MODEL_H = 312
MODEL_W = 312


# ----------------------------
# Helpers
# ----------------------------
def check_cuda(result):
    """
    cuda-python APIs usually return either:
      (error_code, value)
      or just error_code
    """
    if isinstance(result, tuple):
        err = result[0]
        values = result[1:]
    else:
        err = result
        values = ()

    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

    if len(values) == 1:
        return values[0]
    return values


def volume(shape):
    return int(np.prod([int(x) for x in shape]))


# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((MODEL_W, MODEL_H), Image.BILINEAR)

    image = np.array(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # NCHW

    return np.ascontiguousarray(image)


# ----------------------------
# TensorRT engine loading
# ----------------------------
def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)

    with open(path, "rb") as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    return engine


def get_io_names(engine):
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    return input_names, output_names


# ----------------------------
# Buffer allocation
# ----------------------------
def allocate_buffers(engine, context, input_shape):
    input_names, output_names = get_io_names(engine)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected 1 input, found {len(input_names)}: {input_names}")

    input_name = input_names[0]

    # Needed for dynamic-shape engines
    context.set_input_shape(input_name, input_shape)

    stream = check_cuda(cudart.cudaStreamCreate())

    bindings = {}

    # Allocate input
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_size = volume(input_shape)
    input_nbytes = input_size * np.dtype(input_dtype).itemsize

    input_device = check_cuda(cudart.cudaMalloc(input_nbytes))

    bindings[input_name] = {
        "host": None,
        "device": input_device,
        "shape": tuple(input_shape),
        "dtype": input_dtype,
        "nbytes": input_nbytes,
        "is_input": True,
    }

    # Allocate outputs after input shape is set
    for output_name in output_names:
        output_shape = tuple(context.get_tensor_shape(output_name))
        output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

        if any(dim < 0 for dim in output_shape):
            raise RuntimeError(
                f"Output shape for {output_name} is still dynamic: {output_shape}"
            )

        output_size = volume(output_shape)
        output_nbytes = output_size * np.dtype(output_dtype).itemsize

        host_output = np.empty(output_size, dtype=output_dtype)
        device_output = check_cuda(cudart.cudaMalloc(output_nbytes))

        bindings[output_name] = {
            "host": host_output,
            "device": device_output,
            "shape": output_shape,
            "dtype": output_dtype,
            "nbytes": output_nbytes,
            "is_input": False,
        }

    # Bind tensor addresses once
    for name, buf in bindings.items():
        context.set_tensor_address(name, int(buf["device"]))

    return input_name, output_names, bindings, stream


# ----------------------------
# Inference
# ----------------------------
def infer(context, input_name, output_names, bindings, stream, image):
    input_buf = bindings[input_name]

    if image.dtype != input_buf["dtype"]:
        image = image.astype(input_buf["dtype"], copy=False)

    image = np.ascontiguousarray(image)

    if tuple(image.shape) != tuple(input_buf["shape"]):
        raise ValueError(
            f"Input shape mismatch. Expected {input_buf['shape']}, got {image.shape}"
        )

    # Host -> Device
    check_cuda(
        cudart.cudaMemcpyAsync(
            input_buf["device"],
            image.ctypes.data,
            input_buf["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
    )

    # Execute
    ok = context.execute_async_v3(stream_handle=stream)
    if not ok:
        raise RuntimeError("TensorRT execution failed")

    # Device -> Host
    results = []

    for output_name in output_names:
        output_buf = bindings[output_name]

        check_cuda(
            cudart.cudaMemcpyAsync(
                output_buf["host"].ctypes.data,
                output_buf["device"],
                output_buf["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            )
        )

    check_cuda(cudart.cudaStreamSynchronize(stream))

    for output_name in output_names:
        output_buf = bindings[output_name]
        results.append(output_buf["host"].reshape(output_buf["shape"]).copy())

    return results


# ----------------------------
# Benchmark
# ----------------------------
def benchmark(context, input_name, output_names, bindings, stream, image_files, warmup=10):
    if len(image_files) == 0:
        raise ValueError("No images found!")

    sample = preprocess(image_files[0])

    for _ in range(warmup):
        infer(context, input_name, output_names, bindings, stream, sample)

    total_time = 0.0
    count = 0

    for img_path in tqdm.tqdm(image_files):
        image = preprocess(img_path)

        start = timeit.default_timer()
        infer(context, input_name, output_names, bindings, stream, image)
        end = timeit.default_timer()

        total_time += end - start
        count += 1

    return total_time / count


# ----------------------------
# Cleanup
# ----------------------------
def free_buffers(bindings, stream):
    for buf in bindings.values():
        check_cuda(cudart.cudaFree(buf["device"]))

    check_cuda(cudart.cudaStreamDestroy(stream))


# ----------------------------
# Main
# ----------------------------
def main():
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    print("Loading TensorRT engine...")
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    input_shape = (1, 3, MODEL_H, MODEL_W)

    input_name, output_names, bindings, stream = allocate_buffers(
        engine,
        context,
        input_shape,
    )

    print(f"Input: {input_name}, shape={input_shape}")
    print("Outputs:")
    for name in output_names:
        print(f"  {name}: shape={bindings[name]['shape']}, dtype={bindings[name]['dtype']}")

    print("\nRunning TensorRT engine benchmark...")
    trt_time = benchmark(
        context,
        input_name,
        output_names,
        bindings,
        stream,
        image_files,
    )

    print("\n==============================")
    print(f"TensorRT engine avg inference time: {trt_time:.6f} sec")
    print(f"TensorRT engine FPS: {1.0 / trt_time:.2f}")
    print("==============================")

    free_buffers(bindings, stream)


if __name__ == "__main__":
    main()