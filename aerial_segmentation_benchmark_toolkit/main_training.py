from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory
from model_backends import UnetBaselineModelBackend

# limit_memory(memory_limit=5000)


if __name__ == '__main__':
    # Use sample dataset for testing the entire routine
    #dataset = 'dataset-sample'  #  0.5 GB download
    dataset = 'dataset-medium' # 9.0 GB download

    dataset = DroneDeployDataset(dataset, 300, 300).download().generate_chips()
    model_backend = UnetBaselineModelBackend()

    experiment = Experiment("test", dataset, model_backend, batch_size=1)
    experiment.analyze()
    experiment.train(epochs=6)
    experiment.save_model()
    print("Training finished")
