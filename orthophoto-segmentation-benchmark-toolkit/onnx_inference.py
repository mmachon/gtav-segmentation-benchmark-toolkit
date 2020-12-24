import onnxruntime as rt
import numpy
import json
import os
from tqdm import tqdm
from PIL import Image
from timeit import default_timer as timer

WARMUP = 100
INFERENCE_ITERATIONS = 9740

def benchmark_model(model, chip):
    if os.path.isfile(f"inference_benchmarks_2/{model}.json"):
        print(f"{model} already benchmarked")
        return
    sess = rt.InferenceSession(f"./onnx_export/{model}", providers=["CUDAExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    for i in range(WARMUP):
        sess.run([label_name], {input_name: chip})[0]

    inference_timings = []
    for i in tqdm(range(INFERENCE_ITERATIONS)):
        start = timer()
        sess.run([label_name], {input_name: chip})[0]
        end = timer()
        inference_timings.append(end - start)
    print(f"Mean: {numpy.mean(inference_timings)}, STD: {numpy.std(inference_timings)}, MEDIAN: {numpy.median(inference_timings)}, 90q: {numpy.percentile(inference_timings, 90)}")
    scores = {
        "90perc": numpy.percentile(inference_timings, 90),
        "median": numpy.median(inference_timings),
        "mean": numpy.mean(inference_timings),
        "std": numpy.std(inference_timings),
        "timings": inference_timings
    }
    with open(f"inference_benchmarks_2/{model}.json", 'w') as outfile:
        json.dump(scores, outfile)
    return numpy.percentile(inference_timings, 90)

chip = numpy.array(Image.open("dataset-medium/image-chips/test/1d4fbe33f3_F1BE1D4184INSPIRE-000019.png").convert('RGB'))
chip = numpy.expand_dims(numpy.array(chip), axis=0)
chip = chip.astype(numpy.float32)

models = os.listdir("onnx_export")

percentiles = []
for model in models:
    print("Benchmark model", model)
    percentiles.append((benchmark_model(model, chip), model))
with open(f"inference_benchmarks_2/summary.json", 'w') as outfile:
    json.dump(percentiles, outfile)
for percentile in percentiles:
    print(f"{percentile[0]}: {percentile[1]}")

