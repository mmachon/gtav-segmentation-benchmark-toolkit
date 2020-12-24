import onnxruntime as rt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from datasets.dd_dataset_config import LABELMAP_RGB

def chips_from_image(img, size):
    shape = img.shape

    chips = []
    for x in range(0, shape[1], size):
        for y in range(0, shape[0], size):
            chip = img[y:y+size, x:x+size, :]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            chip = np.pad(chip, [(0, y_pad), (0, x_pad), (0, 0)], mode='constant')
            chips.append((chip, x, y))
    return chips

def category2mask(img):
    """ Convert a category image to color mask """
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask


MODEL = "unet_efficientnetb3-2020-11-29_19-37-59"

sess = rt.InferenceSession(f"onnx_export/{MODEL}", providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

images = os.listdir("/mnt/h/edm_big_overlap_50p/")
for image in tqdm(images):
    output_file = f"./predictions/{image[:-4]}-predict.png"
    image = f"/mnt/h/edm_big_overlap_50p/{image}"

    size = 384
    with Image.open(image).convert('RGB') as img:
        nimg = np.array(img)
        shape = nimg.shape
        chips = chips_from_image(nimg, size)

        chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
        prediction = np.zeros(shape[:2], dtype='uint8')

        chip_preds = []
        for chip in np.array([chip for chip, _, _ in chips]):
            chip = np.expand_dims(chip, axis=0)
            predicted_chip = sess.run([label_name], {input_name: chip.astype(np.float32)})[0]
            chip_preds.append(np.squeeze(predicted_chip))
        chip_preds = np.array(chip_preds)

        for (chip, x, y), pred in zip(chips, chip_preds):
            category_chip = np.argmax(pred, axis=-1) + 1
            section = prediction[y:y + size, x:x + size].shape
            prediction[y:y + size, x:x + size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        mask_img = Image.fromarray(mask)
        prediction_overlay_image = Image.blend(img, mask_img, alpha=0.5)
        prediction_overlay_image.save(f"{output_file[:-4]}-overlay.png")
