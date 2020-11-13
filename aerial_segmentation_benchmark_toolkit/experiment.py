import datetime
import zipfile
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from model_analyzer import ModelAnalyzer
from score.scoring import Scoring
from util import *

from datasets.dd_dataset_config import test_ids


class Experiment:

    def __init__(self, title, dataset, model_backend, batch_size, experiment_directory=""):
        if experiment_directory == "":
            self.basedir = os.path.join(f"{os.getcwd()}/experiments",
                                        f"{title}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            os.makedirs(self.basedir)
            os.makedirs(f"{self.basedir}/models")
            os.makedirs(f"{self.basedir}/predictions")
            os.makedirs(f"{self.basedir}/export")
            self.model_backend = model_backend.compile()
        else:
            self.basedir = os.path.join(f"{os.getcwd()}/experiments", experiment_directory)
            self.model_backend = model_backend.load(f"{self.basedir}/models/checkpoint")
        self.tensorboard_log = f"{self.basedir}/tensorboard_logs"
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_analyzer = ModelAnalyzer(self.model_backend)
        self.scoring_backend = Scoring(self.basedir)

    def analyze(self):
        self.dataset.analyze()
        self.model_analyzer.analyze(self.batch_size)
        self.model_analyzer.saveToJSON(self.basedir)

    def train(self, epochs):
        train_data, valid_data = self.dataset.load_dataset(self.batch_size)
        self.model_backend.fit(
            train_data,
            validation_data=valid_data,
            epochs=epochs,
            callbacks=[TensorBoard(
                            log_dir=self.tensorboard_log,
                            histogram_freq=1,
                            write_images=True,
                            update_freq='epoch',
                            profile_batch=2,
                            embeddings_freq=1),
                       ModelCheckpoint(
                           filepath=f"{self.basedir}/models/checkpoint",
                           save_weights_only=True,
                           monitor='val_mIOU',
                           mode='max',
                           save_best_only=True
                       )
                       ]
        )

    def predict_image(self, input_file, output_file):
        size = self.dataset.chip_size
        with Image.open(input_file).convert('RGB') as img:
            nimg = np.array(Image.open(input_file).convert('RGB'))
            shape = nimg.shape
            chips = chips_from_image(nimg, self.dataset.chip_size)

        chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
        prediction = np.zeros(shape[:2], dtype='uint8')
        chip_preds = self.model_backend.predict(np.array([chip for chip, _, _ in chips]), verbose=True)

        for (chip, x, y), pred in zip(chips, chip_preds):
            category_chip = np.argmax(pred, axis=-1) + 1
            section = prediction[y:y + size, x:x + size].shape
            prediction[y:y + size, x:x + size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        Image.fromarray(mask).save(output_file)

    def generate_inference_test_files(self):
        for scene in test_ids:
            imagefile = f'{self.dataset.dataset_name}/images/{scene}-ortho.tif'
            predsfile = os.path.join(self.basedir, f'predictions/{scene}-prediction.png')
            print(f"save to {predsfile}")
            if not os.path.exists(imagefile):
                continue

            print(f'running inference on image {imagefile}.')
            self.predict_image(imagefile, predsfile)

    def score(self):
        self.scoring_backend.score_predictions(self.dataset.dataset_name, self.basedir)

    def benchmark(self):
        # TODO run inference on multiple images and measure time
        pass

    # Generate an easy to evaluate file for later model comparison
    def generate_summary(self):
        pass # TODO

    # Saving model weights of the last epoch
    # Best mIOU related epoch is saved in the checkpoint file
    def save_model(self):
        print("Saving last epoch model")
        self.model_backend.save_weights(f"{self.basedir}/models/last_epoch_model")

    # creating zip directory of experiment models
    def bundle(self):
        print("Creating zip bundle")
        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(f"{self.basedir}/modelbundle.zip", mode='w') as zip_file:
            zip_directory(zip_file, "models", compression, f"{self.basedir}/models")
            zip_directory(zip_file, "tensorboard_logs", compression, f"{self.basedir}/tensorboard_logs")
            zip_file.write(f"{self.basedir}/model_summary.json", "model_summary.json", compress_type=compression)