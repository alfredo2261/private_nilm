import torch.nn as nn
import math
import torch


# class LSTM(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, hidden_size_1, hidden_size_2, fc1, fc2):
#         super(LSTM, self).__init__()
#
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2))
#         # self.conv2 = nn.Conv1d(out_channels, out_channels_2, kernel_size, padding=int((kernel_size-1)/2))
#
#         self.lstm1 = nn.LSTM(out_channels, hidden_size_1, num_layers=2, batch_first=True, dropout=0, bidirectional=True)
#         self.lstm2 = nn.LSTM(2 * hidden_size_1, hidden_size_2, num_layers=2, batch_first=True, dropout=0,
#                              bidirectional=True)
#
#         self.linear1 = nn.Linear(2 * hidden_size_2, fc1)
#         self.linear2 = nn.Linear(fc1, fc2)
#         self.linear3 = nn.Linear(fc2, in_channels)
#
#         self.sigmoid = nn.Sigmoid()
#
#         # self.maxpool = nn.MaxPool1d(kernel_size = kernel_size_2, stride = 1, padding = int((kernel_size_2-1)/2))
#
#         self.dropout = nn.Dropout(p=0.5)
#
#     def forward(self, x):
#         # print("shape0: ", x.shape)
#         out = x.unsqueeze(dim=1)  # This unsqueeze is needed to create the correct shape for conv1 layer
#         out = self.conv1(out)
#         out = self.sigmoid(out)
#         out = self.dropout(out)
#         # out = self.maxpool(out)
#         # print("shape2: ", out.shape)
#         # out = self.relu(out)
#         # out = self.sigmoid(out)
#         # out = self.maxpool(out)
#         # out = self.conv2(out)
#         # out = self.relu(out)
#         # out = self.dropout(out)
#         # print("shape3: ", out.shape)
#         # out, _ = self.lstm1(out)
#         out = out.transpose(1, 2).contiguous()
#         # print("shape3: ", out.shape)
#         out, _ = self.lstm1(out)
#         out, _ = self.lstm2(out)
#         out = self.sigmoid(out)
#         out = self.dropout(out)
#         out = self.linear1(out)
#         out = self.sigmoid(out)
#         out = self.dropout(out)
#         out = self.linear2(out)
#         out = self.sigmoid(out)
#         out = self.dropout(out)
#         out = self.linear3(out)
#         # out = self.sigmoid(out)
#         # print("shape4: ", out.shape)
#         out = out.squeeze()
#         # print("shape5: ", out.shape)
#         return out


class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_size_1, hidden_size_2, fc1):
        super(LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2))
        # self.conv2 = nn.Conv1d(out_channels, out_channels_2, kernel_size, padding=int((kernel_size-1)/2))

        self.lstm1 = nn.LSTM(out_channels, hidden_size_1, num_layers=1, batch_first=True, dropout=0, bidirectional=True)

        self.lstm2 = nn.LSTM(2*hidden_size_1, hidden_size_2, num_layers=1, batch_first=True, dropout=0,
                             bidirectional=True)

        self.attention = nn.MultiheadAttention(embed_dim=2*hidden_size_2, num_heads=10)

        self.linear1 = nn.Linear(2*hidden_size_2, fc1)
        self.linear2 = nn.Linear(fc1, in_channels)
        #self.linear3 = nn.Linear(fc2, in_channels)

        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU()
        # self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(kernel_size = 5, stride = 1, padding = 2)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = x.unsqueeze(dim=1)  # This unsqueeze is needed to create the correct shape for conv1 layer
        out = self.conv1(out)
        out = self.leaky(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous()

        #self.lstm1.flatten_parameters()
        out, _ = self.lstm1(out)
        out = self.leaky(out)
        out = self.dropout(out)

        #self.lstm2.flatten_parameters()
        out, _ = self.lstm2(out)
        out = self.leaky(out)
        out = self.dropout(out)

        out, _ = self.attention(out, out, out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.linear1(out)
        out = self.leaky(out)
        #out = self.dropout(out)

        out = self.linear2(out)
        #out = self.dropout(out)

        out = out.squeeze()
        return out
