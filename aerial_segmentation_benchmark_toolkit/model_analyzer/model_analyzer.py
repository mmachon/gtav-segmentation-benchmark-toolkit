import json
from keras_flops import get_flops

from .util import keras_model_memory_usage_in_bytes


class ModelAnalyzer:

    def __init__(self, model):
        self.model_backend = model
        self.params = None
        self.flops = None
        self.memory_estimation = None

    def analyze(self, batch_size):
        self.calc_params()
        self.calc_flops()
        self.calc_memory_estimation(batch_size)

    def getModelSummary(self):
        return self.model_backend.summary()

    def calc_params(self):
        self.params = self.model_backend.count_params()

    def calc_flops(self):
        self.flops = get_flops(self.model_backend)

    def calc_memory_estimation(self, batch_size):
        self.memory_estimation = keras_model_memory_usage_in_bytes(self.model_backend, batch_size=batch_size)

    def saveToJSON(self, path):
        data = {
            "params": self.params,
            "flops": self.flops,
            "memory_estimation": int(self.memory_estimation)
        }
        with open(f"{path}/model_summary.json", 'w') as outfile:
            json.dump(data, outfile)
