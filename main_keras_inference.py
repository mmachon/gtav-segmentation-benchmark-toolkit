from libs import training_keras
from libs import datasets
from libs import models_keras
from libs import inference_keras
from libs import scoring

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7400)])
  except RuntimeError as e:
    print(e)

import wandb

if __name__ == '__main__':
    #dataset = 'dataset-sample'  #  0.5 GB download
    dataset = 'dataset-medium' # 9.0 GB download

    config = {
        'name' : 'baseline-keras',
        'dataset' : dataset,
    }

    wandb.init(config=config)

    datasets.download_dataset(dataset)

    # use the train model to run inference on all test scenes
    inference_keras.run_inference(dataset, basedir=wandb.run.dir, model_path="./wandb/run-20201105_224508-1aqk414p/files/model-best.h5")

    # scores all the test images compared to the ground truth labels then
    # send the scores (f1, precision, recall) and prediction images to wandb
    score, _ = scoring.score_predictions(dataset, basedir=wandb.run.dir)
    print(score)
    wandb.log(score)
