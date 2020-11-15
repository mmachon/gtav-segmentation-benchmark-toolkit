from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory, enable_dynamic_memory_growth
from model_backends import UnetBaselineModelBackend

from tensorflow.keras import metrics, optimizers
from metrics import CustomMeanIOU

enable_dynamic_memory_growth()

if __name__ == '__main__':
    # Use sample dataset for testing the entire routine
    dataset = 'dataset-sample'  #  0.5 GB download
    #dataset = 'dataset-medium' # 9.0 GB download

    dataset = DroneDeployDataset(dataset, 512).download().generate_chips()
    model_backend = UnetBaselineModelBackend()

    batch_size = 1
    experiment = Experiment("test", dataset, model_backend, batch_size=batch_size,
                            experiment_directory="test-2020-11-15_16-17-46")

    experiment.generate_inference_test_files()
    # experiment.score()
