from tensorflow.keras import metrics
from tensorflow import argmax


class CustomMeanIOU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(argmax(y_true, axis=-1), argmax(y_pred, axis=-1), sample_weight)
