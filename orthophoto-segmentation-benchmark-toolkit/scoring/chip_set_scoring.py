import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, jaccard_score

from .scoring import Scoring
from datasets.dd_dataset_config import INV_LABELMAP


class ChipSetScoring(Scoring):

    def __init__(self, basedir):
        super().__init__(basedir)

    def score_predictions(self, dataset_name):
        test_chips_label = os.listdir(f"./{dataset_name}/label-chips/test")
        test_chips_prediction = os.listdir(f"{self.basedir}/predictions/test-chip-predictions")

        assert (len(test_chips_label) == len(test_chips_prediction))

        test_chip_labels = []
        test_chips_predictions = []
        print("Loading Chips")
        for test_chip_file in test_chips_label:
            chip_id = os.path.basename(test_chip_file)
            test_chip_label = np.array(cv2.imread(f"./{dataset_name}/label-chips/test/{chip_id}"))
            test_chip_prediction = np.array(cv2.imread(f"{self.basedir}/predictions/test-chip-predictions/{chip_id}"))
            for color, category in INV_LABELMAP.items():
                locs = self.wherecolor(test_chip_prediction, color)
                test_chip_prediction[locs] = category - 1
            test_chip_labels.append(test_chip_label)
            test_chips_predictions.append(test_chip_prediction)

        print("Concatenating chips")
        test_chip_label_concat = cv2.hconcat(np.array(test_chip_labels))
        test_chip_prediction_concat = cv2.hconcat(np.array(test_chips_predictions))

        shape = test_chip_label_concat.shape
        test_chip_label_concat = test_chip_label_concat.reshape(shape[0]*shape[1]*shape[2])
        test_chip_prediction_concat = test_chip_prediction_concat.reshape(shape[0] * shape[1] * shape[2])

        print("Calculating precision")
        precision = precision_score(test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print("Calculating recall")
        recall = recall_score(test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print("Calculating IOU")
        jaccard = jaccard_score(test_chip_label_concat, test_chip_prediction_concat, average=None)
        print("Calculating mIOU")
        mean_jaccard = jaccard_score(test_chip_label_concat, test_chip_prediction_concat, average='macro')
        print("Calculating fw_mIOU")
        weighted_mean_jaccard = jaccard_score(test_chip_label_concat, test_chip_prediction_concat, average='weighted')
        print(f'precision={precision} recall={recall}')
        print(f"IOU: {jaccard}")
        print(f"mIOU={mean_jaccard}")
        print(f"fw_mIOU={weighted_mean_jaccard}")

        return precision, recall, jaccard, mean_jaccard, weighted_mean_jaccard
