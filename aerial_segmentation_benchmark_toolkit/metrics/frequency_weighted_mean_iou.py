from tensorflow.keras import metrics


class FrequencyWeightedMeanIOU(metrics.Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
