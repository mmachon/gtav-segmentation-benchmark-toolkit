import sys
import cv2
import os
import numpy as np
import shutil

from .dd_dataset_config import train_ids, val_ids, test_ids, LABELMAP, INV_LABELMAP
from .dataset import Dataset

URLS = {
    'dataset-sample': 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium': 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}


class DroneDeployDataset(Dataset):

    def __init__(self, dataset, chip_size):
        super().__init__(dataset)
        if dataset not in URLS:
            print(f"unknown dataset {dataset}")
            sys.exit(0)
        self.chip_size = chip_size
        self.chip_stride = chip_size

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
        print(f"converting {dataset} image {orthofile} {xsize}x{ysize} to chips ...")

        counter = 0

        for xi in range(0, shape[1] - windowx, stridex):
            for yi in range(0, shape[0] - windowy, stridey):

                orthochip = ortho[yi:yi + windowy, xi:xi + windowx, :]
                labelchip = label[yi:yi + windowy, xi:xi + windowx, :]

                orthochip, classchip = self.color2class(orthochip, labelchip)

                if classchip is None:
                    continue

                orthochip_filename = os.path.join(prefix, 'image-chips', scene + '-' + str(counter).zfill(6) + '.png')
                labelchip_filename = os.path.join(prefix, 'label-chips', scene + '-' + str(counter).zfill(6) + '.png')

                with open(f"{prefix}/{dataset}", mode='a') as fd:
                    fd.write(scene + '-' + str(counter).zfill(6) + '.png\n')

                cv2.imwrite(orthochip_filename, orthochip)
                cv2.imwrite(labelchip_filename, classchip)
                counter += 1
        print(f"Generated {counter} chips")

    def get_split(self, scene):
        if scene in train_ids:
            return "train.txt"
        if scene in val_ids:
            return 'valid.txt'
        if scene in test_ids:
            return 'test.txt'

    def run_chip_generator(self):

        prefix = self.dataset_name
        open(prefix + '/train.txt', mode='w').close()
        open(prefix + '/valid.txt', mode='w').close()
        open(prefix + '/test.txt', mode='w').close()

        if not os.path.exists(os.path.join(prefix, 'image-chips')):
            os.mkdir(os.path.join(prefix, 'image-chips'))

        if not os.path.exists(os.path.join(prefix, 'label-chips')):
            os.mkdir(os.path.join(prefix, 'label-chips'))

        lines = [line for line in open(f'{prefix}/index.csv')]
        lines = list(dict.fromkeys(lines))
        num_images = len(lines) - 1
        print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")

        for lineno, line in enumerate(lines):

            line = line.strip().split(' ')
            scene = line[1]
            dataset = self.get_split(scene)

            if dataset == 'test.txt':
                print(f"not converting test image {scene} to chips, it will be used for inference.")
                continue

            orthofile = os.path.join(prefix, 'images', scene + '-ortho.tif')
            labelfile = os.path.join(prefix, 'labels', scene + '-label.png')

            if os.path.exists(orthofile) and os.path.exists(labelfile):
                print(f"{lineno}/{len(lines)}")
                self.image2tile(prefix, scene, dataset, orthofile, labelfile, self.chip_size, self.chip_size, self.chip_stride, self.chip_stride)