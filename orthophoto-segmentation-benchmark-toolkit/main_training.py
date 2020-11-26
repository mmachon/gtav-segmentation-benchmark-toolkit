import argparse
from experiment import Experiment
from datasets import DroneDeployDataset
from util import *
from model_backends import UnetBackend, PSPnetBackend, FPNBackend, Deeplabv3plusBackend, UnetBaselineModelBackend

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="init dataset", action="store_true")
    parser.add_argument("-t", "--train", help="train model", action="store_true")
    parser.add_argument("-b", "--benchmark", help="run inference", action="store_true")
    parser.add_argument("-s", "--score", help="score model", action="store_true")
    parser.add_argument("-p", "--predict", help="score model", action="store_true")
    args = parser.parse_args()

    config = {
        "experiment_title": "dd_unet_resnet18-2020-11-22_02-46-54",
        "dataset_id": "dataset-medium",
        "chip_size": 384,
        "batch_size": 16,
        "epochs": 40,
        "model_backbone": "mobilenetv3",
        "model_backend": PSPnetBackend,
        "load_experiment": "",
        "load_best_model": True,
    }

    enable_dynamic_memory_growth()

    if args.init:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"]).download().generate_chips()
    else:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"])
    model_backend = config["model_backend"](config["model_backbone"], config["chip_size"])
    experiment = Experiment(config["experiment_title"], dataset, model_backend, batch_size=config["batch_size"],
                            experiment_directory=config["load_experiment"], load_best=config["load_best_model"])

    if args.train:
        experiment.analyze()
        experiment.train(epochs=config["epochs"])
        experiment.save_model()
        experiment.bundle()
        print("Training finished")
        exit()

    if args.benchmark:
        if config["load_experiment"] is None:
            exit("No model selected")
        experiment.benchmark_inference()

    if args.score:
        experiment.score()
        exit()

    if args.predict:
        # TODO
        exit()
