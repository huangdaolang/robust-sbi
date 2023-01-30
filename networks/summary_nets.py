import torch
import torch.nn as nn


class RickerSummary(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(RickerSummary, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.rnn = nn.RNN(input_size, hidden_dim, 1, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_dim, 1, batch_first=True)

        self.conv = nn.Sequential(nn.Conv1d(self.input_size, self.hidden_dim, 3, 3),
                                  nn.Conv1d(4, self.hidden_dim, 3, 3),
                                  nn.Conv1d(4, self.hidden_dim, 3, 3),
                                  nn.AvgPool1d(3))

    def forward(self, Y):
        embeddings_conv = self.conv(Y.reshape(-1, 1, 100)).reshape(-1, 100, 4)

        stat_conv = torch.mean(embeddings_conv, dim=1)
        return embeddings_conv, stat_conv


class LotkaSummary(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LotkaSummary, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_dim, 1, batch_first=True)

        self.conv = nn.Sequential(nn.Conv1d(self.input_size, self.hidden_dim, 3, 3),
                                  nn.Conv1d(4, self.hidden_dim, 3, 3),
                                  nn.Conv1d(4, self.hidden_dim, 3, 3),
                                  nn.AvgPool1d(3))

    def forward(self, Y):
        batch_size = Y.size(0)
        hidden, c = self.init_hidden(batch_size)
        out, (embeddings_lstm, c) = self.lstm(Y, (hidden, c))


    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden, c