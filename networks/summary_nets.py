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


class OUPSummary(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(OUPSummary, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_size = input_size
        self.num_layers = 1
        self.lstm = nn.LSTM(1, self.hidden_dim, self.num_layers, batch_first=True)

        self.conv = nn.Sequential(nn.Conv1d(self.input_size, self.hidden_dim, 3, 2),
                                  nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, 2),
                                  nn.Conv1d(4, self.hidden_dim, 3, 3),
                                  nn.AvgPool1d(3))
        self.fc = nn.Linear(self.hidden_dim, 4)

    def forward(self, Y):
        current_device = Y.device
        batch_size = Y.size(0)

        #         embeddings_conv = self.conv(Y.reshape(-1, 1, 40)).reshape(-1, 100, 4)
        #         stat_conv = torch.mean(embeddings_conv, dim=1)

        hidden, c = self.init_hidden(100 * batch_size, current_device)
        out, (embeddings_lstm, c) = self.lstm(Y.reshape(100 * batch_size, 50, 1), (hidden, c))

        embeddings_lstm = embeddings_lstm.reshape(batch_size, 100, self.hidden_dim)

        stat_lstm = torch.mean(embeddings_lstm, dim=1)
        out = self.fc(stat_lstm)

        #         stat = torch.cat([stat_conv, stat_lstm], dim=1)

        return embeddings_lstm, out

    def init_hidden(self, batch_size, current_device):
        hidden = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(current_device)
        c = torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).to(current_device)
        return hidden, c