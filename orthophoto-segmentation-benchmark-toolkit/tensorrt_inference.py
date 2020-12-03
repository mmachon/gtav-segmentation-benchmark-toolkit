import onnx
import onnx_tensorrt.backend as backend
import numpy as np
from PIL import Image
import os

model = onnx.load("/tensorrt_models/onnx_graph")
engine = backend.prepare(model, device='CUDA:1')

chip_file_list = [f"dataset-medium/image-chips/test/{chip_file}" for chip_file in os.listdir(
    f"dataset-medium/image-chips/test")]
chip_files = [(np.array(Image.open(chip_file).convert('RGB'))) for chip_file in chip_file_list[1191:]]
chip = chip_files[0]
chip = np.expand_dims(np.array(chip), axis=0)
input_data = chip.astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)
