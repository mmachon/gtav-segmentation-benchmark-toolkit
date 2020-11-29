import os
from multiprocessing import Pool
import cv2
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score, jaccard_score

from .scoring import Scoring
from datasets.dd_dataset_config import INV_LABELMAP as DD_INV_LABELMAP
from datasets.potsdam_config import INV_LABELMAP as POTSDAM_INV_LABELMAP
from datasets.potsdam_config import LABELMAP as POTSDAM_LABELMAP


class PotsdamScoring(Scoring):

    def __init__(self, basedir):
        super().__init__(basedir)

    def score_image(self, test_file):
        print("Scoring started")
        label_class = test_file[0].reshape((test_file[0].shape[0] * test_file[0].shape[1]) * 3)
        pred_class = test_file[1].reshape((test_file[1].shape[0] * test_file[1].shape[1] * 3))
        precision = precision_score(label_class, pred_class, average='weighted')
        recall = recall_score(label_class, pred_class, average='weighted')
        jaccard = jaccard_score(label_class, pred_class, average=None)
        mean_jaccard = jaccard_score(label_class, pred_class, average='macro')
        weighted_mean_jaccard = jaccard_score(label_class, pred_class, average='weighted')
        print(precision, recall, jaccard, mean_jaccard, weighted_mean_jaccard)
        return precision, recall, jaccard, mean_jaccard, weighted_mean_jaccard

    def score_predictions(self, dataset_name):
        test_chips_label = os.listdir(f"./{dataset_name}/5_Labels_class")
        test_chips_prediction = os.listdir(f"{self.basedir}/predictions/potsdam")

        assert (len(test_chips_label) == len(test_chips_prediction))

        test_file_list = []
        print("Loading images")
        for test_files in test_chips_label:
            print("Loading " + test_files)
            test_file_id = os.path.basename(test_files)
            test_file_label = np.array(cv2.imread(f"./{dataset_name}/5_Labels_class/{test_file_id}"))
            test_file_prediction = np.array(cv2.imread(f"{self.basedir}/predictions/potsdam/{test_file_id[:-9]}RGB-prediction.png"))

            # COMBINED LABELS: [BUILDING, CLUTTER, VEGETATION, GROUND, CAR]

            for color, category in POTSDAM_INV_LABELMAP.items():
                locs = self.wherecategory(test_file_label, category-1)
                if category == 1:
                    test_file_label[locs] = 0
                elif category == 2:
                    test_file_label[locs] = 1
                elif category == 3:
                    test_file_label[locs] = 3
                elif category == 4:
                    test_file_label[locs] = 2
                elif category == 5:
                    test_file_label[locs] = 3
                elif category == 6:
                    test_file_label[locs] = 4
            for color, category in DD_INV_LABELMAP.items():
                locs = self.wherecolor(test_file_prediction, color)
                if category == 1:
                    test_file_prediction[locs] = 0
                elif category == 2:
                    test_file_prediction[locs] = 1
                elif category == 3:
                    test_file_prediction[locs] = 2
                elif category == 4:
                    test_file_prediction[locs] = 1
                elif category == 5:
                    test_file_prediction[locs] = 3
                elif category == 6:
                    test_file_prediction[locs] = 4
            zipped_test_files = (test_file_label, test_file_prediction)
            test_file_list.append(zipped_test_files)

        precision = []
        recall = []
        jaccard = []
        mean_jaccard = []
        weighted_mean_jaccard = []
        results = Parallel(n_jobs=3)(
            delayed(self.score_image)(test_file) for test_file in test_file_list
        )
        print("Calculate stats")
        for result in results:
            precision.append(result[0])
            recall.append(result[1])
            jaccard.append(result[2])
            mean_jaccard.append(result[3])
            weighted_mean_jaccard.append(result[4])
        class_jaccard = np.array(jaccard).transpose()

        # Compute test set scores
        scores_means = {
            'pr_mean': np.mean(precision),
            'pr_std': np.std(precision),
            're_mean': np.mean(recall),
            're_std': np.std(recall),
            'mj_mean': np.mean(mean_jaccard),
            'mj_std': np.std(mean_jaccard),
            'wmj_mean': np.mean(weighted_mean_jaccard),
            'wmj_std': np.std(weighted_mean_jaccard),
            'building_jc_mean': np.mean(class_jaccard[0]),
            'building_jc_std': np.std(class_jaccard[0]),
            'clutter_jc_mean': np.mean(class_jaccard[1]),
            'clutter_jc_std': np.std(class_jaccard[1]),
            'vegetation_jc_mean': np.mean(class_jaccard[2]),
            'vegetation_jc_std': np.std(class_jaccard[2]),
            'ground_jc_mean': np.mean(class_jaccard[3]),
            'ground_jc_std': np.std(class_jaccard[3]),
            'car_jc_mean': np.mean(class_jaccard[4]),
            'car_jc_std': np.std(class_jaccard[4]),
        }

        score_dict = {
            "precision": precision,
            "recall": recall,
            "mean_jaccard": mean_jaccard,
            "weighted_mean_jaccard": weighted_mean_jaccard,
            "jaccard_classwise": class_jaccard.tolist(),
            "score_means": scores_means
        }

        print(score_dict)

        return score_dict
