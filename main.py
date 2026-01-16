import torch
from random import randint


class Model:
    def __init__(self, size):
        self.size = size
        self.w = list()

        for i in range(1, len(size)):
            self.w.append(torch.randn(self.size[i], self.size[i-1] + 1) * 0.1)

    def act(self, z):
        return torch.tanh(z)

    def df(self, z):
        s = self.act(z)
        return 1 - s * s

    def forward(self, x):
        value = list()
        act = list()
        act.append(x)
        act_with_bias = list()

        for i in range(len(self.w)):
            act_with_bias.append(torch.cat([act[i], torch.tensor([1.0])]))
            value.append(torch.matmul(self.w[i], act_with_bias[i]))
            act.append(self.act(value[i]))

        return act, act_with_bias, value

    def backward(self, target, act, act_with_bias, value):
        delta = list()
        delta.append((act[-1] - target) * self.df(value[-1]))

        for i in range(len(self.w) - 1):
            w_i = self.w[-1 - i]
            w_i_without_bias = w_i[:, :-1]

            delta.append(torch.matmul(w_i_without_bias.t(), delta[-1]) * self.df(value[-2 - i]))

        return delta

    def fit(self, x_train, y_train, n, lmd):
        for epoch in range(n):
            k = randint(0, len(y_train) - 1)

            act, act_with_bias, value = self.forward(x_train[k])
            delta = self.backward(y_train[k], act, act_with_bias, value)

            if epoch % 5000 == 0:
                print(f"\nЭпоха {epoch}:")
                print(f"  Вход: {x_train[k]}")
                print(f"  Выход: {act[-1]}, Цель: {y_train[k]}")
                print(f"  Ошибка: {(act[-1] - y_train[k])}")
                print(f"  Количество дельт: {len(delta)}")
                print(f"  Дельта[0] (выходной слой): {delta[0]}")
                print(f"  Веса W[0][0,:3] до обновления: {self.w[0][0, :3]}")

            for i in range(len(self.w)):
                self.w[i] = self.w[i] - lmd * torch.outer(delta[-1 - i], act_with_bias[i])

torch.manual_seed(1)

size = (3, 8, 4, 2)
a = Model(size)

x_train = torch.FloatTensor([
    [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
    [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
])

y_train = torch.FloatTensor([[-1, 1], [1, -1], [1, -1], [-1, 1], [-1, 1], [1, -1], [1, -1], [-1, 1]])
lmd = 0.1
n = 20000

a.fit(x_train, y_train, n, lmd)

for x, d in zip(x_train, y_train):
   act, act_with_bias, value = a.forward(x)
   y = act[-1]

   print(f"Выходное значение НС: {y} => {d}")