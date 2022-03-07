import random
from clean_data import create_activations
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lstm import LSTM
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PecanStreetDataset_train(Dataset):

    def __init__(self, path, appliance, window_length, buildings):
        self.x, self.y = create_activations(
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


class PecanStreetDataset_test(Dataset):

    def __init__(self, path, appliance, window_length, buildings):
        self.x, self.y = create_activations(
            path,
            appliance,
            window_length,
            buildings)

        self.n_samples = self.x.shape[0]

        self.x, self.y = zip(*random.sample(list(zip(self.x, self.y)), round(0.5 * self.n_samples)))

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        self.n_samples = self.x.shape[0]

        self.x = torch.from_numpy(self.x)

        self.y = torch.from_numpy(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class PecanStreetDataset_val(Dataset):

    def __init__(self, path, appliance, window_length, buildings):
        self.x, self.y = create_activations(
            path,
            appliance,
            window_length,
            buildings)

        self.n_samples = self.x.shape[0]

        self.x, self.y = zip(*random.sample(list(zip(self.x, self.y)), round(0.5 * self.n_samples)))

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

        self.n_samples = self.x.shape[0]

        self.x = torch.from_numpy(self.x)

        self.y = torch.from_numpy(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def make_train_data(config, train_data, appliance, window_length, train_buildings):
    root_path = r"C:\Users\aar245.CORNELL\Desktop\Fall2021_new\ithaca_Real_2019\1min_real_"

    train_dataset = PecanStreetDataset_train(str(root_path) + str(train_data) + "2019.csv", appliance, window_length,
                                             train_buildings)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=0)

    return train_loader


def make_test_val_data(config, test_data, appliance, window_length, test_buildings):
    root_path = r"C:\Users\aar245.CORNELL\Desktop\Fall2021_new\ithaca_Real_2019\1min_real_"

    validation_dataset = PecanStreetDataset_val(str(root_path) + str(test_data) + "2019.csv", appliance, window_length,
                                                test_buildings)
    test_dataset = PecanStreetDataset_test(str(root_path) + str(test_data) + "2019.csv", appliance, window_length,
                                           test_buildings)

    validation_loader = DataLoader(dataset=validation_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   num_workers=0)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=0)

    return validation_loader, test_loader


def make_model(config):
    model = LSTM(
        config.in_channels,
        config.out_channels,
        config.kernel_size,
        config.hidden_size_1,
        config.hidden_size_2,
        config.fc1,
        config.fc2).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config.learning_rate,
        weight_decay = 0.021543979275305797)
        #weight_decay=0.00005)

    return model, criterion, optimizer
