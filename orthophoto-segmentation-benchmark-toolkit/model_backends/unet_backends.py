from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from .segmentation_models import Unet

from .model_backend import ModelBackend


class UnetBackend(ModelBackend):

    def __init__(self, backbone, chip_size=320):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        mobilenet_min = self.backbone in ["mobilenetv3_minimalistic", "mobilenetv3small_minimalistic"]
        model_backend = Unet(self.backbone, input_shape=(self.chip_size, self.chip_size, 3), classes=6, activation="softmax", minimalistc=mobilenet_min)
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        model_backend.summary()
        plot_model(model_backend, to_file="unetmobile2.png", show_shapes=True)
        return model_backend
