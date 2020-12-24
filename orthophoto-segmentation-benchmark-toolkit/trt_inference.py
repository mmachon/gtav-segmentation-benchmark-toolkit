import tensorflow as tf
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from tqdm import tqdm
from util import *
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import json
limit_memory(500)


class TensorRTBenchmark:
    def __init__(self):
        chip_file_list = [f"dataset-medium/image-chips/test/{chip_file}" for chip_file in os.listdir(
            f"dataset-medium/image-chips/test")]
        print(f"Loading {len(chip_file_list)} chips")
        self.chip_files = [(np.array(Image.open(chip_file).convert('RGB')), os.path.basename(chip_file)) for chip_file in tqdm(chip_file_list[1191:])]

    def benchmark_tensorrt_inference(self, model):
        # Loading TensorRT model
        print(f"Loading model {model}")
        frozen_func = convert_variables_to_constants_v2(tf.saved_model.load(
            f"tensorrt_models/{model}", tags=[tag_constants.SERVING]).signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY])

        inference_timings = []
        print("Warmup")
        for i in range(10):
            chip, chip_id = self.chip_files[0]
            chip = np.expand_dims(np.array(chip), axis=0)
            frozen_func(tf.constant(chip.astype(np.float32)))

        # Run benchmark
        for i in tqdm(range(1000)):
            chip, chip_id = self.chip_files[0]
            chip = np.expand_dims(np.array(chip), axis=0)
            chip = tf.constant(chip.astype(np.float32))
            start = timer()
            frozen_func(chip)[0].numpy()
            end = timer()
            inference_timings.append(end - start)
        print(f"Mean: {np.mean(inference_timings)}, STD: {np.std(inference_timings)}, MEDIAN: {np.median(inference_timings)}")
        benchmark_summary = {"timings": inference_timings,
                       "mean": np.mean(inference_timings),
                       "std": np.std(inference_timings),
                       "median": np.median(inference_timings),
                       "90_perc": np.percentile(inference_timings, 90)}
        with open(f"model_scores/{model}-inference_benchmark.json", "w") as inference_json:
            json.dump(benchmark_summary, inference_json)


tensorrt_benchmark = TensorRTBenchmark()
model_list = ["TFTRT_FP16_psp_efficientnetb4-2020-11-29_00-33-32-saved_model"]
for model in model_list:
    tensorrt_benchmark.benchmark_tensorrt_inference(model)
