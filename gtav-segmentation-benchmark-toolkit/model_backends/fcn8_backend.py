from .model_backend import ModelBackend
from .amazing_semantic_segmentation.models.fcn import FCN
from tensorflow.keras import optimizers, losses


class FCN8Backend(ModelBackend):

    def __init__(self, backbone, chip_size):
        super().__init__(chip_size)
        self.backbone = "resnet50"

    def compile(self):
        super().compile()
        model_backend = FCN(6, base_model="ResNet50")(input_size=(self.chip_size, self.chip_size))
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=self.metrics
        )
        return model_backend