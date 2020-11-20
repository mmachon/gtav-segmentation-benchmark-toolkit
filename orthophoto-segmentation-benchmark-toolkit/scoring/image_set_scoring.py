import os
import cv2
import json

from .scoring import Scoring
from datasets.dd_dataset_config import INV_LABELMAP, test_ids

from sklearn.metrics import precision_score, recall_score, jaccard_score, classification_report
import numpy as np


class ImageSetScoring(Scoring):

    def __init__(self, basedir):
        super().__init__(basedir)

    def score_masks(self, labelfile, predictionfile, scene):

        label = cv2.imread(labelfile)
        prediction = cv2.imread(predictionfile)

        shape = label.shape[:2]

        label_class = np.zeros(shape, dtype='uint8')
        pred_class = np.zeros(shape, dtype='uint8')

        for color, category in INV_LABELMAP.items():
            locs = self.wherecolor(label, color)
            label_class[locs] = category

        for color, category in INV_LABELMAP.items():
            locs = self.wherecolor(prediction, color)
            pred_class[locs] = category

        label_class = label_class.reshape((label_class.shape[0] * label_class.shape[1]))
        pred_class = pred_class.reshape((pred_class.shape[0] * pred_class.shape[1]))

        # Remove all predictions where there is a IGNORE (magenta pixel) in the groud label and then shift labels down 1 index
        not_ignore_locs = np.where(label_class != 0)
        label_class = label_class[not_ignore_locs] - 1
        pred_class = pred_class[not_ignore_locs] - 1

        print(classification_report(label_class, pred_class, [0, 1, 2, 3, 4, 5]))
        precision = precision_score(label_class, pred_class, average='weighted')
        recall = recall_score(label_class, pred_class, average='weighted')
        jaccard = jaccard_score(label_class, pred_class, average=None)
        mean_jaccard = jaccard_score(label_class, pred_class, average='macro')
        weighted_mean_jaccard = jaccard_score(label_class, pred_class, average='weighted')
        print(f'precision={precision} recall={recall}')
        print(f"IOU: {jaccard}")
        print(f"mIOU={mean_jaccard}")
        print(f"fw_mIOU={weighted_mean_jaccard}")

        return precision, recall, jaccard, mean_jaccard, weighted_mean_jaccard

    def score_predictions(self, dataset):

        precision = []
        recall = []
        jaccard = []
        mean_jaccard = []
        weighted_mean_jaccard = []

        predictions = []
        confusions = []

        for i, scene in enumerate(test_ids):
            labelfile = f'{dataset}/labels/{scene}-label.png'
            predsfile = os.path.join(self.basedir, f"predictions/{scene}-prediction.png")

            if not os.path.exists(labelfile):
                continue

            if not os.path.exists(predsfile):
                continue

            print(f"Calculating score {i}/{len(test_ids)} for {scene}")

            prec, rec, jac, m_jac, w_jac, savefile = self.score_masks(labelfile, predsfile, scene)

            precision.append(prec)
            recall.append(rec)
            jaccard.append(jac)
            mean_jaccard.append(m_jac)
            weighted_mean_jaccard.append(w_jac)

            predictions.append(predsfile)
            confusions.append(savefile)

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
            'water_jc_mean': np.mean(class_jaccard[3]),
            'water_jc_std': np.std(class_jaccard[3]),
            'ground_jc_mean': np.mean(class_jaccard[4]),
            'ground_jc_std': np.std(class_jaccard[4]),
            'car_jc_mean': np.mean(class_jaccard[5]),
            'car_jc_std': np.std(class_jaccard[5]),
        }

        score_dict = {
            "precision": precision,
            "recall": recall,
            "mean_jaccard": mean_jaccard,
            "weighted_mean_jaccard": weighted_mean_jaccard,
            "jaccard_classwise": class_jaccard.tolist(),
            "score_means": scores_means
        }

        scoring_file_path = f"{self.basedir}/scoring_summary.json"
        print(f"Writing scores to {scoring_file_path}")
        with open(scoring_file_path, 'w') as outfile:
            json.dump(score_dict, outfile)
