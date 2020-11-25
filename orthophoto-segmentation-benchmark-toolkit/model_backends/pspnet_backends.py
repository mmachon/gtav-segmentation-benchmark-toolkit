from tensorflow.keras import optimizers
from .segmentation_models import PSPNet

from .model_backend import ModelBackend


class PSPnetBackend(ModelBackend):

    def __init__(self, backbone, chip_size=320):
        super().__init__(chip_size)
        if backbone not in self.available_backbones:
            print("Backbone not found")
            raise ValueError
        self.backbone = backbone

    def compile(self):
        super().compile()
        model_backend = PSPNet(self.backbone, input_shape=(self.chip_size, self.chip_size, 3), classes=6, activation="sigmoid")
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        return model_backend
