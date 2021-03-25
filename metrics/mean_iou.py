from tensorflow.keras import metrics
from tensorflow import argmax


class CustomMeanIOU(metrics.MeanIoU):
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(CustomMeanIOU, self).__init__(num_classes=6, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(argmax(y_true, axis=-1), argmax(y_pred, axis=-1), sample_weight)
