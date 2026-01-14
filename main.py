import torch
from random import randint


class Model:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.w1 = torch.randn(hidden_size1, input_size + 1) * 0.1
        self.w2 = torch.randn(hidden_size2, hidden_size1 + 1, ) * 0.1
        self.w3 = torch.randn(output_size, hidden_size2 + 1, ) * 0.1

    def act(self, z):
        return torch.tanh(z)

    def df(self, z):
        s = self.act(z)
        return 1 - s * s

    def forward(self, x):
        x_bias = torch.cat([x, torch.tensor([1.0])])

        z1 = torch.mv(self.w1, x_bias)
        s1 = self.act(z1)
        s1_bias = torch.cat([s1, torch.tensor([1.0])])

        z2 = torch.mv(self.w2, s1_bias)
        s2 = self.act(z2)
        s2_bias = torch.cat([s2, torch.tensor([1.0])])

        z3 = torch.mv(self.w3, s2_bias)
        y = self.act(z3)

        return y, z3, s2, z2, s1, z1

    def backward(self, target, y, z3, z2, z1):
        e = y - target
        delta3 = e * self.df(z3)
        delta2 = torch.mv(self.w3[:, :-1].t(), delta3, ) * self.df(z2)
        delta1 = torch.mv(self.w2[:, :-1].t(), delta2) * self.df(z1)

        return delta1, delta2, delta3

    def fit(self, x_train, y_train, n, lmd):
        for _ in range(n):
            k = randint(0, len(y_train) - 1)

            y, z3, s2, z2, s1, z1 = self.forward(x_train[k])

            delta1, delta2, delta3 = self.backward(y_train[k], y, z3, z2, z1)

            s2_bias = torch.cat([s2, torch.tensor([1.0])])
            s1_bias = torch.cat([s1, torch.tensor([1.0])])
            x_bias = torch.cat([x_train[k], torch.tensor([1.0])])

            self.w3 = self.w3 - lmd * torch.outer(delta3, s2_bias)
            self.w2 = self.w2 - lmd * torch.outer(delta2, s1_bias)
            self.w1 = self.w1 - lmd * torch.outer(delta1, x_bias)

torch.manual_seed(1)

a = Model(3, 8, 4, 1)

x_train = torch.FloatTensor([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
])

y_train = torch.FloatTensor([-1, 1, 1, -1, -1, 1, 1, -1])
lmd = 0.1
n = 20000

a.fit(x_train, y_train, n, lmd)

for x, d in zip(x_train, y_train):
   y, z3, s2, z2, s1, z1 = a.forward(x)
   print(f"Выходное значение НС: {y} => {d}")