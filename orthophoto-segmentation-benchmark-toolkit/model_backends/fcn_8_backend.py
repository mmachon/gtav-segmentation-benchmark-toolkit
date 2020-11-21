from tensorflow.keras import optimizers
from .amazing_semantic_segmentation.models import FCN

from .segmentation_models_backend import ModelBackend


class FCN8Backend(ModelBackend):

    def __init__(self, chip_size=320):
        super().__init__(chip_size)

    def compile(self):
        model_backend = FCN(6, version="FCN-8s", base_model="ResNet50")(input_size=(self.chip_size, self.chip_size))
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        return model_backend
