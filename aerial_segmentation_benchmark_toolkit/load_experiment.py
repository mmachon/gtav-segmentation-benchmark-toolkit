from experiment import Experiment
from datasets import DroneDeployDataset
from util import limit_memory
from model_backends import build_unet

from tensorflow.keras import metrics, optimizers
from metrics import CustomMeanIOU

limit_memory(memory_limit=6500)


if __name__ == '__main__':
    # Use sample dataset for testing the entire routine
    #dataset = 'dataset-sample'  #  0.5 GB download
    dataset = 'dataset-medium' # 9.0 GB download

    dataset = DroneDeployDataset(dataset, 300, 300).download().generate_chips()
    model_backend = build_unet(encoder='resnet50')

    model_backend.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss='categorical_crossentropy',
        metrics=[
            metrics.Precision(top_k=1, name='precision'),
            metrics.Recall(top_k=1, name='recall'),
            CustomMeanIOU(num_classes=6, name='mIOU'),
        ]
    )

    batch_size = 1
    experiment = Experiment("test", dataset, model_backend, batch_size=batch_size, experiment_directory="test-2020-11-12_20-30-34")

    experiment.generate_inference_test_files()
    experiment.score()
