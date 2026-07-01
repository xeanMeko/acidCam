import onnxruntime as ort
import tqdm
import os
import timeit
import numpy as np
from PIL import Image
import cv2

nano_path = "exports/rfdetr/rfdetr-seg-nano.onnx"
large_path = "exports/rfdetr/rfdetr-seg-large.onnx"
session = ort.InferenceSession(nano_path)

image_folder = "test/"

MODEL_H = 312
MODEL_W = 312

def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((MODEL_W, MODEL_H), Image.BILINEAR)

    image = np.array(image, dtype=np.float32) / 255.0

    # Common RF-DETR/ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # CHW -> NCHW
    return np.ascontiguousarray(image)

def infer(image):
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session.run(output_names, {input_name: image})

# use timeit to measure the inference time
total_time = 0
num_images = 0
for image_name in tqdm.tqdm(os.listdir(image_folder)):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue  # Skip non-image files

    image_path = os.path.join(image_folder, image_name)

    start_time = timeit.default_timer()

    image = preprocess(image_path)
    infer(image)

    end_time = timeit.default_timer()
    total_time += end_time - start_time
    num_images += 1

print(f"Average inference time: {total_time / num_images:.4f} seconds")