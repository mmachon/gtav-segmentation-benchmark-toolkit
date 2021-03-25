from tensorflow.keras import optimizers
from .deeplabv3plus import Deeplabv3

from .model_backend import ModelBackend


class Deeplabv3plusBackend(ModelBackend):

    def __init__(self, backbone="xception", chip_size=384):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        model_backend = Deeplabv3(weights='cityscapes', input_shape=(self.chip_size, self.chip_size, 3), classes=6, backbone=self.backbone, OS=16, activation="softmax")
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        return model_backend
