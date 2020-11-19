import sys
import cv2
import os
import random
import numpy as np
from skimage import morphology
import shutil
import json
from multiprocessing import Pool

from .util import closest_color


from .dd_dataset_config import train_ids, val_ids, test_ids, LABELMAP, INV_LABELMAP
from .dataset import Dataset

URLS = {
    'dataset-sample': 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium': 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}


class DroneDeployDataset(Dataset):

    def __init__(self, dataset, chip_size, worker=8):
        super().__init__(dataset)
        if dataset not in URLS:
            print(f"unknown dataset {dataset}")
            sys.exit(0)
        self.chip_size = chip_size
        self.chip_stride = chip_size
        self.coverage = {}
        self.worker = worker

    def download(self):
        """ Download a dataset, extract it and create the tiles """

        filename = f'{self.dataset_name}.tar.gz'
        url = URLS[self.dataset_name]
        if not os.path.exists(filename):
            print(f'downloading dataset "{self.dataset_name}"')
            os.system(f'curl "{url}" -o {filename}')
        else:
            print(f'zipfile "{filename}" already exists, remove it if you want to re-download.')

        if not os.path.exists(self.dataset_name):
            print(f'extracting "{filename}"')
            os.system(f'tar -xvf {filename}')
        else:
            print(f'folder "{self.dataset_name}" already exists, remove it if you want to re-create.')
        return self

    def generate_chips(self, reset_chips=False):
        image_chips = f'{self.dataset_name}/image-chips'
        label_chips = f'{self.dataset_name}/label-chips'
        if reset_chips:
            shutil.rmtree(image_chips)
            shutil.rmtree(label_chips)
        if not os.path.exists(image_chips) and not os.path.exists(label_chips):
            print("creating chips")
            self.run_chip_generator()
        else:
            print(f'chip folders "{image_chips}" and "{label_chips}" already exist, remove them to recreate chips.')
        return self

    def analyze(self):
        pass # TODO compute class label distribution for each chip set


    def color2class(self, orthochip, img):
        ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
        ret = np.dstack([ret, ret, ret])
        colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

        # Skip any chips that would contain magenta (IGNORE) pixels
        seen_colors = set([tuple(color) for color in colors])
        IGNORE_COLOR = LABELMAP[0]
        if IGNORE_COLOR in seen_colors:
            return None, None

        for color in colors:
            if tuple(color) not in INV_LABELMAP:
                available_colors = np.array(list(INV_LABELMAP.keys()))
                color = closest_color(available_colors, [color])[0]
            locs = np.where((img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]))
            ret[locs[0], locs[1], :] = INV_LABELMAP[tuple(color)] - 1

        return orthochip, ret

    def image2tile(self, prefix, scene, dataset, orthofile, labelfile, windowx, windowy, stridex, stridey):

        ortho = cv2.imread(orthofile)
        label = cv2.imread(labelfile)

        assert (ortho.shape[0] == label.shape[0])
        assert (ortho.shape[1] == label.shape[1])

        shape = ortho.shape

        xsize = shape[1]
        ysize = shape[0]

        counter = 0

        for xi in range(0, shape[1] - windowx, stridex):
            for yi in range(0, shape[0] - windowy, stridey):

                orthochip = ortho[yi:yi + windowy, xi:xi + windowx, :]
                labelchip = label[yi:yi + windowy, xi:xi + windowx, :]

                orthochip, classchip = self.color2class(orthochip, labelchip)

                if classchip is None:
                    continue

                if random.random() < 0.66:
                    orthochip_filename = os.path.join(prefix, 'image-chips/training', scene + '-' + str(counter).zfill(6) + '.png')
                    labelchip_filename = os.path.join(prefix, 'label-chips/training', scene + '-' + str(counter).zfill(6) + '.png')
                elif random.random() < 0.50:
                    orthochip_filename = os.path.join(prefix, 'image-chips/validation', scene + '-' + str(counter).zfill(6) + '.png')
                    labelchip_filename = os.path.join(prefix, 'label-chips/validation', scene + '-' + str(counter).zfill(6) + '.png')
                else:
                    orthochip_filename = os.path.join(prefix, 'image-chips/test', scene + '-' + str(counter).zfill(6) + '.png')
                    labelchip_filename = os.path.join(prefix, 'label-chips/test', scene + '-' + str(counter).zfill(6) + '.png')


                with open(f"{prefix}/{dataset}", mode='a') as fd:
                    fd.write(scene + '-' + str(counter).zfill(6) + '.png\n')

                cv2.imwrite(orthochip_filename, orthochip)
                cv2.imwrite(labelchip_filename, classchip)
                counter += 1
        chip_coverage = counter * (windowx*windowy) / (xsize*ysize)
        return counter, chip_coverage

    def get_split(self, scene):
        if scene in train_ids:
            return "train.txt"
        if scene in val_ids:
            return 'valid.txt'
        if scene in test_ids:
            return 'test.txt'

    def run_chip_generator(self):

        if self.dataset_name=="dataset-medium":
            if not os.path.isdir("./dataset-medium/cleaned-labels"):
                self.enhance_dataset_medium()

        prefix = self.dataset_name
        open(prefix + '/train.txt', mode='w').close()
        open(prefix + '/valid.txt', mode='w').close()
        open(prefix + '/test.txt', mode='w').close()

        if not os.path.exists(os.path.join(prefix, 'image-chips')):
            os.mkdir(os.path.join(prefix, 'image-chips'))
            os.mkdir(os.path.join(prefix, 'image-chips/training'))
            os.mkdir(os.path.join(prefix, 'image-chips/validation'))
            os.mkdir(os.path.join(prefix, 'image-chips/test'))

        if not os.path.exists(os.path.join(prefix, 'label-chips')):
            os.mkdir(os.path.join(prefix, 'label-chips'))
            os.mkdir(os.path.join(prefix, 'label-chips/training'))
            os.mkdir(os.path.join(prefix, 'label-chips/validation'))
            os.mkdir(os.path.join(prefix, 'label-chips/test'))

        lines = [line for line in open(f'{prefix}/index.csv')]
        lines = list(dict.fromkeys(lines))
        num_images = len(lines) - 1
        print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")

        pool = Pool(processes=self.worker)
        results = [result for result in pool.map(self.convertImage, lines) if result]

        for scene, coverage in results:
            self.coverage[scene] = coverage

        with open(f"{prefix}/{self.chip_size}-chip_coverage-pol.json", 'w') as outfile:
            json.dump(self.coverage, outfile)

        self.analyze()

    def convertImage(self, line):
        line = line.strip().split(' ')
        scene = line[1]
        dataset = self.get_split(scene)

        label_dir = "cleaned-labels" if os.path.isdir(f"{self.dataset_name}/cleaned-labels") else "labels"
        orthofile = os.path.join(self.dataset_name, 'images', scene + '-ortho.tif')
        labelfile = os.path.join(self.dataset_name, label_dir, scene + '-label.png')
        if os.path.exists(orthofile) and os.path.exists(labelfile):
            chip_count, scene_coverage = self.image2tile(self.dataset_name, scene, dataset, orthofile, labelfile, self.chip_size,
                                             self.chip_size, self.chip_stride, self.chip_stride)
            print(f"Processed {scene} -> Generated {chip_count} chips with a image coverage of {round(scene_coverage*100,2)}%")
            return scene, scene_coverage

    def enhance_dataset_medium(self):
        label_files = []
        for file in os.listdir("./dataset-medium/labels"):
            label_files.append(file)
        if not os.path.isdir("./dataset-medium/cleaned-labels"):
            os.makedirs("./dataset-medium/cleaned-labels")

        for i, file in enumerate(label_files):
            print(f"Processing {i + 1}/{len(label_files)}")
            img = cv2.imread("./dataset-medium/labels/" + file)

            ignoreLower = np.array([255, 0, 255], dtype="uint8")
            ignoreUpper = np.array([255, 0, 255], dtype="uint8")

            ignoreMask = cv2.inRange(img, ignoreLower, ignoreUpper)
            imglab = morphology.label(ignoreMask)  # create labels in segmented image
            cleaned = morphology.remove_small_objects(imglab, min_size=256, connectivity=2)
            mask = (imglab - cleaned)

            cleaned_img = cv2.inpaint(img, np.uint8(mask), 3, cv2.INPAINT_TELEA)
            cv2.imwrite("./dataset-medium/cleaned-labels/" + file, cleaned_img)