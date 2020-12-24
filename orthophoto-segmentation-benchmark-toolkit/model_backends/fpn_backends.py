from tensorflow.keras import optimizers
from .segmentation_models import FPN

from .model_backend import ModelBackend


class FPNBackend(ModelBackend):

    def __init__(self, backbone, chip_size=320):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        super().compile()
        mobilenet_min = self.backbone in ["mobilenetv3_minimalistic", "mobilenetv3small_minimalistic"]
        model_backend = FPN(self.backbone, input_shape=(1152, 1152, 3), classes=6, activation="softmax", encoder_weights=None)
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        model_backend.summary()
        return model_backend
