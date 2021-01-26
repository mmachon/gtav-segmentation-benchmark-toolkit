from tensorflow.keras import optimizers, losses
from .segmentation_models import PSPNet

from .model_backend import ModelBackend


class PSPnetBackend(ModelBackend):

    def __init__(self, backbone, chip_size=384):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        super().compile()
        model_backend = PSPNet(self.backbone, input_shape=(self.chip_size, self.chip_size, 3), classes=6, activation="softmax", downsample_factor=16)
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=self.metrics
        )
        return model_backend
