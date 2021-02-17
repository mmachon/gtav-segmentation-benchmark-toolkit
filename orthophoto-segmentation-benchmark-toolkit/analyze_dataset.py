from datasets import DroneDeployDataset


if __name__ == '__main__':

    dataset = 'dataset-medium'

    dataset = DroneDeployDataset(dataset, 384).download().generate_chips()
    train_distribution = dataset.analyze("./dataset-medium/label-chips/training")
    valid_distribution = dataset.analyze("./dataset-medium/label-chips/validation")
    test_distribution = dataset.analyze("./dataset-medium/label-chips/test")

    print(train_distribution, valid_distribution, test_distribution)