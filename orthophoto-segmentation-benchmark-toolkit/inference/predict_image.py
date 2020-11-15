import numpy as np
from PIL import Image
from util import *


def generate_predict_image(basedir, input_file, scene, model, chip_size):
    output_file = os.path.join(basedir, f'predictions/{scene}-prediction.png')
    size = chip_size
    with Image.open(input_file).convert('RGB') as img:
        nimg = np.array(Image.open(input_file).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg, chip_size)

        chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
        prediction = np.zeros(shape[:2], dtype='uint8')

        chip_preds = []
        for chip in np.array([chip for chip, _, _ in chips]):
            np.expand_dims(chip, axis=0)
            predicted_chip = model.predict(np.expand_dims(np.array(chip), axis=0))
            chip_preds.append(np.squeeze(predicted_chip))
        chip_preds = np.array(chip_preds)

        for (chip, x, y), pred in zip(chips, chip_preds):
            category_chip = np.argmax(pred, axis=-1) + 1
            section = prediction[y:y + size, x:x + size].shape
            prediction[y:y + size, x:x + size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        mask_img = Image.fromarray(mask)
        mask_img.save(output_file)
        prediction_overlay_image = Image.blend(img, mask_img, alpha=0.6)
        prediction_overlay_image.save(f"{output_file[:-4]}-overlay.png")
        generate_prediction_chips(np.array(prediction_overlay_image), chip_size, basedir, scene)


def generate_prediction_chips(prediction_overlay_image, chip_size, basedir, scene):
    chips = chips_from_image(prediction_overlay_image, chip_size)
    for i, chip in enumerate(chips):
        Image.fromarray(chip[0]).save(f"{basedir}/predictions/chips/{scene}-{i}.png")
