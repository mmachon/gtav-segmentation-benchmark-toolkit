from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory, enable_dynamic_memory_growth
from model_backends import UnetBaselineModelBackend
from model_backends.unet_backends import UnetBackend

from tensorflow.keras import metrics, optimizers
from metrics import CustomMeanIOU

enable_dynamic_memory_growth()

if __name__ == '__main__':
    dataset = 'dataset-medium'  # 9.0 GB download

    dataset = DroneDeployDataset(dataset, 320).download().generate_chips()
    model_backend = UnetBackend("resnet50")

    batch_size = 1
    experiment = Experiment("test", dataset, model_backend, batch_size=batch_size,
                            experiment_directory="test-2020-11-18_14-20-23", load_best=True)

    experiment.benchmark_inference()
