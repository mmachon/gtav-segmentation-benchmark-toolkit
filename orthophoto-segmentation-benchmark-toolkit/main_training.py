import argparse

from experiment import Experiment
from datasets import DroneDeployDataset
from util import *
from model_backends import *

enable_dynamic_memory_growth()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="model weights of given eperiment will be used for training", default=30)
    parser.add_argument("--bs", help="model weights of given eperiment will be used for training", default=4)
    parser.add_argument("--experiment", help="model weights of given eperiment will be used for training", default="")
    args = parser.parse_args()


    dataset = 'dataset-sample' # 9.0 GB download
    size = 512

    dataset = DroneDeployDataset(dataset, size).download().generate_chips()
    model_backend = UnetBackend('efficientnetb3')

    experiment = Experiment("test", dataset, model_backend, batch_size=args.bs,
                            experiment_directory=args.experiment, load_best=False)
    experiment.analyze()
    experiment.train(epochs=args.epochs)
    experiment.save_model()
    experiment.bundle()
    print("Training finished")
