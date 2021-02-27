from .model_backend import ModelBackend
from .amazing_semantic_segmentation.models.bisegnet import get_model
from tensorflow.keras import optimizers, losses


class BisegnetBackend(ModelBackend):

    def __init__(self, backbone, chip_size):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        super().compile()
        model_backend = get_model(backbone_name=self.backbone)
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=self.metrics
        )
        return model_backend
