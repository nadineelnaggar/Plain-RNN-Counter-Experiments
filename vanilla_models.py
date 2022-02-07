import torch
import torch.nn as nn



device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.output_size=output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h0):

        x, h0 = self.lstm(x, h0)
        x = self.sigmoid(x).view(-1, self.output_size)
        return x

    def init_hidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, 1, self.hidden_dim).to(device))


