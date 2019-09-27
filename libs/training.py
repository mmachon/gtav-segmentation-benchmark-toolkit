"""
    train.py - Sample implementation of a Dynamic Unet using FastAI
    2019 - Nicholas Pilkington, DroneDeploy
"""

from fastai.vision import *
from fastai.callbacks.hooks import *
from libs import inference
from libs import scoring
from libs.util import MySaveModelCallback, ExportCallback, MyCSVLogger, Precision, Recall, FBeta
from libs import datasets

import wandb
from wandb.fastai import WandbCallback


def train_model(dataset):
    """ Trains a DynamicUnet on the dataset """

    epochs = 20
    lr = 1e-4
    size = 300
    wd = 1e-2
    bs = 8 # reduce this if you are running out of GPU memory
    pretrained = True

    config = {
        'epochs' : epochs,
        'lr' : lr,
        'size' : size,
        'wd' : wd,
        'bs' : bs,
        'pretrained' : pretrained,
    }

    wandb.config.update(config)

    metrics = [
        Precision(average='weighted', clas_idx=1),
        Recall(average='weighted', clas_idx=1),
        FBeta(average='weighted', beta=1, clas_idx=1),
    ]

    data = datasets.load_dataset(dataset, size, bs)
    encoder_model = models.resnet18
    learn = unet_learner(data, encoder_model, metrics=metrics, wd=wd, bottle=True, pretrained=pretrained, callback_fns=WandbCallback)

    callbacks = [
        MyCSVLogger(learn, filename='example_model'),
        ExportCallback(learn, "example_model", monitor='f_beta'),
        MySaveModelCallback(learn, every='epoch', monitor='f_beta')
    ]

    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr, callbacks=callbacks)
