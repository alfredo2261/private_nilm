import torch.nn as nn
import math
import torch


class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_size_1, hidden_size_2, fc1, batch_size, window_size):
        super(LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8)
        self.conv3 = nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6)
        self.conv4 = nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5)
        self.conv5 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5)

        self.linear1 = nn.Linear(in_features=50*(window_size-29), out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=1)

        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        # self.tanh = nn.Tanh()

        self.dropout = nn.Dropout(p=0.2)
        self.batch_size = batch_size

    def forward(self, x):
        out = x.unsqueeze(dim=1)  # This unsqueeze is needed to create the correct shape for conv1 layer
        out = self.conv1(out)
        out = self.leaky(out)
        #out = self.dropout(out)

        out = self.conv2(out)
        out = self.leaky(out)
        #out = self.dropout(out)

        out = self.conv3(out)
        out = self.leaky(out)
        #out = self.dropout(out)

        out = self.conv4(out)
        out = self.leaky(out)
        out = self.dropout(out)

        out = self.conv5(out)
        out = self.leaky(out)
        out = self.dropout(out)
        out = out.view(x.shape[0], -1)

        out = self.linear1(out)
        out = self.leaky(out)
        out = self.dropout(out)

        out = self.linear2(out)

        return out
