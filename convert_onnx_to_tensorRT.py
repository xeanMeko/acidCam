import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)

network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("exports/rfdetr/rfdetr-seg-nano.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

engine = builder.build_serialized_network(network, config)

with open("exports/rfdetr/rfdetr-seg-nano.engine", "wb") as f:
    f.write(engine)