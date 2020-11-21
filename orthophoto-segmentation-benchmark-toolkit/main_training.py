import argparse

from experiment import Experiment
from datasets import DroneDeployDataset
from util import *
from model_backends import UnetBackend, PSPnetBackend, FPNBackend

enable_dynamic_memory_growth()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="epochs", default=30)
    parser.add_argument("--bs", help="batch_size", default=4)
    parser.add_argument("--experiment", help="model weights of given eperiment will be used for training", default="")
    args = parser.parse_args()

    # SET DATASET
    dataset_id = 'dataset-medium' # 9.0 GB download

    # SET CHIP SIZE
    size = 320

    #SET MODEL
    backbone = 'mobilenetv3small'
    model_backend = PSPnetBackend(backbone, size).compile().summary()

    dataset = DroneDeployDataset(dataset_id, size).download().generate_chips()
    experiment = Experiment("test", dataset, model_backend, batch_size=args.bs,
                            experiment_directory=args.experiment, load_best=False)
    experiment.analyze()
    experiment.train(epochs=args.epochs)
    experiment.save_model()
    experiment.bundle()
    print("Training finished")
