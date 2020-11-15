import datetime
import zipfile
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.backend import clear_session
from model_analyzer import ModelAnalyzer
from score.scoring import Scoring
from inference.predict_image import *

from datasets.dd_dataset_config import test_ids


class Experiment:

    def __init__(self, title, dataset, model_backend, batch_size, experiment_directory="", load_best=False):
        if experiment_directory == "":
            self.experiment_title = f"{title}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.basedir = os.path.join(f"{os.getcwd()}/experiments", self.experiment_title)
            self.init_experiment_directory_structure()
            self.model_backend = model_backend.compile()
        else:
            self.basedir = os.path.join(f"{os.getcwd()}/experiments", experiment_directory)
            print("Loading existing model")
            if load_best:
                self.model_backend = model_backend.load(f"{self.basedir}/models/checkpoint")
            else:
                self.model_backend = model_backend.load(f"{self.basedir}/models/last_epoch_model")
        self.tensorboard_log = f"{self.basedir}/tensorboard_logs"
        self.dataset = dataset
        self.batch_size = batch_size
        self.model_analyzer = ModelAnalyzer(self.model_backend)
        self.scoring_backend = Scoring(self.basedir)

    def init_experiment_directory_structure(self):
        os.makedirs(self.basedir)
        os.makedirs(f"{self.basedir}/models")
        os.makedirs(f"{self.basedir}/predictions")
        os.makedirs(f"{self.basedir}/export")

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
                            profile_batch='500,510',
                            embeddings_freq=1,),
                       ModelCheckpoint(
                           filepath=f"{self.basedir}/models/checkpoint",
                           save_weights_only=True,
                           monitor='val_mIOU',
                           mode='max',
                           save_best_only=True
                       )
                       ]
        )

    def generate_inference_test_files(self):
        clear_session()
        for scene in test_ids:
            imagefile = f'{self.dataset.dataset_name}/images/{scene}-ortho.tif'
            predsfile = os.path.join(self.basedir, f'predictions/{scene}-prediction.png')
            if not os.path.exists(imagefile):
                continue
            print(f'running inference on image {imagefile}.')
            generate_predict_image(imagefile, predsfile, self.model_backend, self.dataset.chip_size)

    def score(self):
        self.scoring_backend.score_predictions(self.dataset.dataset_name, self.basedir)

    def benchmark(self):
        # TODO run inference on multiple images and measure time
        pass

    # Generate an easy to evaluate file for later model comparsion
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
        with zipfile.ZipFile(f"{self.basedir}/{self.experiment_title}.zip", mode='w') as zip_file:
            zip_directory(zip_file, "models", compression, f"{self.basedir}/models")
            zip_directory(zip_file, "tensorboard_logs", compression, f"{self.basedir}/tensorboard_logs")
            zip_file.write(f"{self.basedir}/model_summary.json", "model_summary.json", compress_type=compression)