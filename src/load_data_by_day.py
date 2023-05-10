import random
from clean_data_seq2point import create_activations
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lstm_seq2point import LSTM
import torch.nn as nn

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

