from abc import ABC, abstractmethod
import numpy as np


class Scoring(ABC):

    def __init__(self, basedir):
        self.basedir = basedir

    @abstractmethod
    def score_predictions(self, dataset):
        pass

    def wherecolor(self, img, color, negate=False):

        k1 = (img[:, :, 0] == color[0])
        k2 = (img[:, :, 1] == color[1])
        k3 = (img[:, :, 2] == color[2])

        if negate:
            return np.where(not (k1 & k2 & k3))
        else:
            return np.where(k1 & k2 & k3)