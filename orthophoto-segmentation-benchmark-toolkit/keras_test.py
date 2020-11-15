from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory, enalbe_dynamic_memory_growth
from model_backends import UnetBaselineModelBackend

enalbe_dynamic_memory_growth()

# Testing whole pipeline
if __name__ == '__main__':
    # Use sample dataset for testing the entire routine
    dataset = 'dataset-sample'  #  0.5 GB download
    # dataset = 'dataset-medium' # 9.0 GB download

    dataset = DroneDeployDataset(dataset, chip_size=512)\
        .download()\
        .generate_chips()
    model_backend = UnetBaselineModelBackend()

    experiment = Experiment("test", dataset, model_backend, batch_size=1,
                            experiment_directory="test-2020-11-14_18-31-49")
    experiment.analyze()
    experiment.train(epochs=5)
    experiment.save_model()
    experiment.generate_inference_test_files()
    experiment.score()
    experiment.bundle()
    print("Experiment finished")
