from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory
from model_backends import UnetBaselineModelBackend

# limit_memory(memory_limit=5000)


if __name__ == '__main__':
    dataset = 'dataset-medium' # 9.0 GB download
    size = 512

    dataset = DroneDeployDataset(dataset, size).download().generate_chips()
    model_backend = UnetBaselineModelBackend()

    experiment = Experiment("test", dataset, model_backend, batch_size=1, experiment_directory="test-2020-11-14_18-31-49")
    experiment.analyze()
    experiment.train(epochs=6)
    experiment.save_model()
    experiment.bundle()
    print("Training finished")
