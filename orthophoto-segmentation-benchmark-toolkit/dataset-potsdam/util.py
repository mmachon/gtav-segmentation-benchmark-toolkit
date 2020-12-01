from datasets.potsdam_config import INV_LABELMAP as POTSDAM_INV_LABELMAP
from datasets.potsdam_config import LABELMAP as POTSDAM_LABELMAP

import numpy as np
import os
import cv2
from tqdm import tqdm

def color2class(img):
    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    ret = np.dstack([ret, ret, ret])
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    # Skip any chips that would contain magenta (IGNORE) pixels
    seen_colors = set([tuple(color) for color in colors])
    IGNORE_COLOR = POTSDAM_LABELMAP[0]
    if IGNORE_COLOR in seen_colors:
        return None, None

    for color in colors:
        locs = np.where((img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]))
        if tuple(color) not in POTSDAM_INV_LABELMAP:
            ret[locs[0], locs[1], :] = 4
        else:
            ret[locs[0], locs[1], :] = POTSDAM_INV_LABELMAP[tuple(color)] - 1
    return ret

def to_class():
    files = os.listdir("./5_Labels_all")
    for img in files:
        print("Converting: " + img)
        test_file_id = os.path.basename(img)
        if os.path.isfile("./2_Ortho_class/" + test_file_id):
            continue
        test_file_label = np.array(cv2.imread(f"./5_Labels_all/{test_file_id}"))
        cv2.imwrite("./2_Ortho_class/" + test_file_id, color2class(test_file_label))


def resize(dir):
    files = os.listdir(f"./{dir}")
    for img_file in tqdm(files):
        img = cv2.imread(f"./{dir}/{img_file}")
        resized_img = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"./{dir}_gsd10/{img_file}", resized_img)



resize("2_Ortho_RGB")