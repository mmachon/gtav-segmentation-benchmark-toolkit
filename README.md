# GTA V Segmentation Benchmark Toolkit

This repository provides a toolkit to run semantic segmentation benchmarks on several state of the art neuronal networks. It is based on *DroneDeploy Machine Learning Segmentation Benchmark* and *Orthophoto Segmentation Benchmark Toolkit*. It was enhanced by specific functionalities in context of my master thesis. Furthermore a GTA V dataset was implemented to train neuronal networks on syntethic data and observe their suitability on real datasets like the DroneDeploy dataset or C2Land dataset.

# Prerequisites

* Python 3.7 <
* CUDA 10.1 for GPU accelerated training and inference

# Installing
Clone the repository

Install python prerequisites
```bash
python3 -m pip install requirements.txt
```

Choose dataset (dataset_id) and initialize dataset by executing 
```bash
python3 main_interface.py -i
```

# How to use

Models are defined by the config dictionary in main_training.py:

```python
    config = {
        "experiment_title": "test",             # choose a experiment title
        "dataset_id": "dataset-gta-1280",       # select a dataset, currently supported: [dataset-gta-1280, dataset-gta-3200, 
                                                # dataset-c2land, dataset-medium, dataset-sample]
        "chip_size": 192,                       # choose chip_size to generate from images and labels
        "batch_size": 8,                        # choose number of batch_size
        "epochs": 40,                           # choose number of epochs
        "model_backbone": "efficientnetb2",     # select encoder
        "model_backend": FPNBackend,            # select architechture
        "load_experiment": "",                  # experiment title of the model to load 
        "load_best_model": True,                # decide whether to use the best or last epoch model of the loaded experiment
        "prediction_postprocessing": False,     # decide wheter to use smooth tile prediction in
                                                # --score_generalisation and --predict
    }
```

## Train

To train a model with given config execute

```bash
python3 main_interface.py -t

```

A new directory at experiments/ will be created where the model and additional files like a model summary and plots of the train/validation loss and mIoU will be placed after training finished 

## Predict

## Score

To evalutate the model performance generate predictions of the test chip and start scroing by executing the command

```bash
python3 main_interface.py -b -s

```
This will create a inference_benchmark.json file inference timings and a scores.json with precision, recall, class-vise IoU, mIoU and frequency weighted mIoU will be created 

## Export

```bash
python3 main_interface.py -b -s

```

will export the model to tensorflow's and onnx format
