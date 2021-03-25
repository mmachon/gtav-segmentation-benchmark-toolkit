import argparse
from experiment import Experiment
from datasets import DroneDeployDataset
from util import *
from model_backends import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="init dataset", action="store_true")
    parser.add_argument("-t", "--train", help="train model", action="store_true")
    parser.add_argument("-b", "--benchmark", help="run inference", action="store_true")
    parser.add_argument("-s", "--score", help="score model", action="store_true")
    parser.add_argument("-sg", "--score_generalisation", help="score model generalisation", action="store_true")
    parser.add_argument("-p", "--predict", help="predict image", action="store_true")
    parser.add_argument("-e", "--export", help="export model", action="store_true")
    args = parser.parse_args()

    config = {

        "experiment_title": "testgen",          # choose a experiment title
        "dataset_id": "dataset-gta-1280",       # select a dataset, currently supported: [dataset-gta-1280, dataset-gta-3200, dataset-medium, dataset-sample]
        "chip_size": 384,                       # choose chip_size to generate from images and labels
        "batch_size": 8,                        # choose number of batch_size
        "epochs": 40,                           # choose number of epochs
        "model_backbone": "efficientnetb2",     # select encoder
        "model_backend": FPNBackend,            # select architechture
        "load_experiment": "",                  # experiment title of the model to load
        "load_best_model": True,                # decide whether to use the best or last epoch model of the loaded experiment
    }

    enable_dynamic_memory_growth()

    if args.init:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"]).download().generate_chips()
    else:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"])

    if config["model_backbone"]:
        model_backend = config["model_backend"](config["model_backbone"], config["chip_size"])
    else:
        model_backend = config["model_backend"](config["chip_size"])
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

    if args.score_generalisation:
        experiment.score_generalization()

    if args.predict:
        experiment.predict("./RGB100MP_2020-06-02_10-05-18cropsq.tif", "./testpp4.png", postprocessing=True)

    if args.export:
        experiment.export_model()
