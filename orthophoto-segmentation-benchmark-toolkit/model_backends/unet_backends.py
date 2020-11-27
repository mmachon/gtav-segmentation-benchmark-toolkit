from tensorflow.keras import optimizers
from .segmentation_models import Unet

from .model_backend import ModelBackend


class UnetBackend(ModelBackend):

    def __init__(self, backbone, chip_size=320):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        model_backend = Unet(self.backbone, input_shape=(self.chip_size, self.chip_size, 3), classes=6, activation="softmax")
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        return model_backend
