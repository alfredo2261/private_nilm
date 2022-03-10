import random
from clean_data_seq2point import create_activations
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lstm_seq2point import LSTM
import torch.nn as nn


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset(Dataset):

    def __init__(self, path, appliance, window_length, buildings):
        self.x, self.y, self.y_std, self.y_mean = create_activations(
            path,
            appliance,
            window_length,
            buildings)

        self.n_samples = self.x.shape[0]

        self.x = torch.from_numpy(self.x)

        self.y = torch.from_numpy(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class PecanSlice(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def make_train_data(config, train_data, appliance, window_length, train_buildings):
    root_path = "/home/Alfredo/input/1min_real_"
    train_dataset = PecanStreetDataset(str(root_path) + str(train_data) + "2019.csv", appliance, window_length,
                                             train_buildings)

    train_seq_std = train_dataset.y_std
    train_seq_mean = train_dataset.y_mean

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=0)

    return train_loader, train_seq_std, train_seq_mean


def make_test_val_data(config, test_data, appliance, window_length, test_buildings):
    root_path = "/home/Alfredo/input/1min_real_"

    test_validation_dataset = PecanStreetDataset(str(root_path) + str(test_data) + "2019.csv", appliance, window_length,
                                                test_buildings)
    seq_std = test_validation_dataset.y_std
    seq_mean = test_validation_dataset.y_mean

    #num_examples = round(0.5*len(test_validation_dataset))

    #validation_dataset = random.sample(list(test_validation_dataset), num_examples)
    #test_dataset = random.sample(list(test_validation_dataset), num_examples)

    validation_dataset = test_validation_dataset[0:round(0.5*len(test_validation_dataset))]
    validation_dataset = PecanSlice(validation_dataset[0], validation_dataset[1])
    #validation_dataset = np.array([(validation_dataset[0][i], validation_dataset[1][i]) for i in range(len(validation_dataset[0]))])
    test_dataset = test_validation_dataset[round(0.5*len(test_validation_dataset)):]
    #test_dataset = np.array([(test_dataset[0][i], test_dataset[1][i]) for i in range(len(test_dataset[0]))])
    test_dataset = PecanSlice(test_dataset[0], test_dataset[1])
    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=0)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=0)

    return validation_loader, test_loader, seq_std, seq_mean


def make_model(config):
    model = LSTM(
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        config.hidden_size_1,
        config.hidden_size_2,
        config.fc1,
        config.batch_size,
        config.window_size
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay = config.weight_decay)

    return model, criterion, optimizer
