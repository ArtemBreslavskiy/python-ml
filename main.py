import torch
from random import randint


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = torch.FloatTensor(input_size + 1, hidden_size).uniform_(-0.5, 0.5)
        self.W2 = torch.FloatTensor(hidden_size + 1, output_size).uniform_(-0.5, 0.5)

    def forward(self, x):
        x = torch.cat([x, torch.tensor([1])])