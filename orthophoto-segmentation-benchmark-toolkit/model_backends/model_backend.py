from abc import ABC, abstractmethod
from tensorflow.keras import metrics
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from metrics import CustomMeanIOU


class ModelBackend(ABC):

    def __init__(self, chip_size):
        self.available_backbones = ["resnet50", "efficientnetb3", "mobilenetv3", "mobilenetv3small", "efficientnetb0"]
        self.chip_size = chip_size
        self.metrics = [
                metrics.Precision(top_k=1, name='precision'),
                metrics.Recall(top_k=1, name='recall'),
                CustomMeanIOU(num_classes=6, name='mIOU'),
            ]

    def load(self, weights_file_path):
        model_backend = self.compile()
        model_backend.load_weights(weights_file_path)
        return model_backend

    @abstractmethod
    def compile(self):
        pass
