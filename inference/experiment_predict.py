import json
import os
import numpy as np
from experiment import Experiment

def experiment_predict(config, experiment):
    imagePostprocessing = config["prediction_postprocessing"]
    inference_timings = []

    print("Starting Prediction")
    dataset_path = f"./{config.get('dataset_id')}/images"
    if not os.path.isdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch"):
        os.mkdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch")
    if not os.path.isdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/predictions"):
        os.mkdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/predictions")
    if not os.path.isdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/overlay-predictions"):
        os.mkdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/overlay-predictions")
    if not os.path.isdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/smooth-predictions"):
        os.mkdir(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch/smooth-predictions")

    dirList=os.listdir(dataset_path)
    for filename in dirList:
        inference_timing = experiment.predict(f"{dataset_path}/{filename}", f"{filename[:-10]}", postprocessing=imagePostprocessing)
        inference_timings.append(inference_timing)
        
    jsonPrefix = "smooth_" if imagePostprocessing==True else ""
    with open(f"./experiments/{config.get('load_experiment')}/predictions/{config.get('dataset_id')}_{str(config.get('chip_size'))}ch//prediction_{jsonPrefix}inference.json", "w") as inference_json:
        json.dump({"timings": inference_timings,
                "mean": np.mean(inference_timings),
                "std": np.std(inference_timings),
                "median": np.median(inference_timings),
                "90_perc": np.percentile(inference_timings, 90)}, inference_json)