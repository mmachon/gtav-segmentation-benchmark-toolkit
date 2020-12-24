import argparse
from experiment import Experiment
from datasets import DroneDeployDataset
from util import *
from model_backends import *
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="init dataset", action="store_true")
    parser.add_argument("-t", "--train", help="train model", action="store_true")
    parser.add_argument("-b", "--benchmark", help="run inference", action="store_true")
    parser.add_argument("-s", "--score", help="score model", action="store_true")
    parser.add_argument("-sg", "--score_generalisation", help="score model generalisation", action="store_true")
    parser.add_argument("-p", "--predict", help="predict image", action="store_true")
    parser.add_argument("-pd", "--predict_dir", help="predict images from direcotry", action="store_true")
    parser.add_argument("-e", "--export", help="export model", action="store_true")
    args = parser.parse_args()

    config = {
        "experiment_title": "unet",
        "dataset_id": "dataset-medium",
        "chip_size": 384,
        "batch_size": 8,
        "epochs": 1,
        "model_backbone": "efficientnetb2",
        "model_backend": FPNBackend,
        "load_experiment": "fpn_efficientnetb2-2020-11-30_11-29-55",
        "load_best_model": True,
    }

    enable_dynamic_memory_growth()

    if args.init:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"]).download().generate_chips()
    else:
        dataset = DroneDeployDataset(config["dataset_id"], config["chip_size"])
    if config["model_backbone"] == "":
        model_backend = config["model_backend"](config["chip_size"])
    else:
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

    if args.score_generalisation:
        experiment.score_generalization()

    if args.predict:
        experiment.predict("./ortho_002005update.tif.png", "update", postprocessing=False)

    if args.predict_dir:
        images = os.listdir("/mnt/h/edm_big_overlap_50p/")[:120]
        for image in tqdm(images):
            experiment.predict(f"/mnt/h/edm_big_overlap_50p/{image}", image, postprocessing=False)

    if args.export:
        experiment.export_model()
