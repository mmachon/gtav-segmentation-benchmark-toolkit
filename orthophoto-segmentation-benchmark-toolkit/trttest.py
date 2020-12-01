import tensorflow as tf
import matplotlib
import cv2
import os
from tensorflow.keras import metrics
from tensorflow import argmax
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from tqdm import tqdm
from util import *
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import json
from sklearn.metrics import precision_score, recall_score, jaccard_score
enable_dynamic_memory_growth()


# TODO -> OOD
class TensorRTBenchmark:
    def __init__(self, score=True):
        chip_file_list = [f"dataset-medium/image-chips/test/{chip_file}" for chip_file in os.listdir(
            f"dataset-medium/image-chips/test")]
        label_file_list = [f"dataset-medium/label-chips/test/{chip_file}" for chip_file in os.listdir(
            f"dataset-medium/image-chips/test")]
        print(f"Loading {len(chip_file_list)} chips")
        self.chip_files = [(np.array(Image.open(chip_file).convert('RGB')), os.path.basename(chip_file)) for chip_file in tqdm(chip_file_list)]

        if score:
            chip_files = []
            print("Concate test chips for scoring")
            for test_chip_file in tqdm(label_file_list):
                test_chip_label = np.array(cv2.imread(test_chip_file))
                chip_files.append(np.amax(test_chip_label, axis=2))
            self.test_chip_label_concat = cv2.hconcat(np.array(chip_files))
            shape = self.test_chip_label_concat.shape
            self.test_chip_label_concat = self.test_chip_label_concat.reshape(shape[0] * shape[1])

        self.predictions = []

    def benchmark_and_score(self, model):
        self.benchmark_tensorrt_inference(model)
        self.score_predictions(model)

    def benchmark_tensorrt_inference(self, model):
        # Loading TensorRT model
        print(f"Loading model {model}")
        saved_model_loaded = tf.saved_model.load(
            f"tensorrt_models/{model}", tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        frozen_func = convert_variables_to_constants_v2(graph_func)


        inference_timings = []
        print("Warmup")
        for i in range(50):
            chip, chip_id = self.chip_files[0]
            chip = np.expand_dims(np.array(chip), axis=0)
            frozen_func(tf.constant(chip.astype(np.float32)))

        # Run benchmark
        for chip, chip_id in tqdm(self.chip_files):
            chip = np.expand_dims(np.array(chip), axis=0)
            start = timer()
            predicted_chip = frozen_func(tf.constant(chip.astype(np.float32)))[0].numpy()
            end = timer()
            inference_timings.append(end - start)
            self.predictions.append((np.squeeze(predicted_chip), chip_id))
        print(f"Mean: {np.mean(inference_timings)}, STD: {np.std(inference_timings)}, MEDIAN: {np.median(inference_timings)}")
        benchmark_summary = {"timings": inference_timings,
                       "mean": np.mean(inference_timings),
                       "std": np.std(inference_timings),
                       "median": np.median(inference_timings),
                       "90_perc": np.percentile(inference_timings, 90)}
        with open(f"model_scores/{model}-inference_benchmark.json", "w") as inference_json:
            json.dump(benchmark_summary, inference_json)

    def score_predictions(self, model):
        category_chips = []
        print("Mapping prediction chips to class")
        for prediction, chip_id in tqdm(self.predictions):
            category_chip = np.argmax(prediction, axis=-1)
            category_chips.append(category_chip)

        test_chip_prediction_concat = cv2.hconcat(np.array(category_chips))
        test_chip_prediction_concat = test_chip_prediction_concat.reshape(self.test_chip_label_concat.size)
        print("Calculating precision")
        precision = precision_score(self.test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print("Calculating recall")
        recall = recall_score(self.test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print("Calculating IOU")
        jaccard = jaccard_score(self.test_chip_label_concat, test_chip_prediction_concat, average=None)
        print("Calculating mIOU")
        mean_jaccard = jaccard_score(self.test_chip_label_concat, test_chip_prediction_concat, average='macro')
        print("Calculating fw_mIOU")
        weighted_mean_jaccard = jaccard_score(self.test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print(f'precision={precision} recall={recall}')
        print(f"IOU: {jaccard}")
        print(f"mIOU={mean_jaccard}")
        print(f"fw_mIOU={weighted_mean_jaccard}")

        scores = {"precision": precision,
                "recall": recall,
                "iou": jaccard.tolist(),
                "m_iou:": mean_jaccard,
                "fwm_iou": weighted_mean_jaccard
                }

        with open(f"model_scores/{model}-score.json", "w") as score_json:
            json.dump(scores, score_json)


tensorrt_benchmark = TensorRTBenchmark()
model_list = ["TFTRT_FP16_psp_efficientnetb4-2020-11-29_00-33-32-saved_model"]
for model in model_list:
    tensorrt_benchmark.benchmark_and_score(model)
