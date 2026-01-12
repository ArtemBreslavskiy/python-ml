import torch


def act(x):
    return 0 if x < 0.5 else 1


def go(house, rock, attr):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor([house, rock, attr], dtype=torch.float32, device=device)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]], device=device)
    Wout = torch.tensor(([-1.0, 1.0]), device=device)

    Zh = torch.mv(Wh, X)

    Uh = torch.tensor([act(x) for x in Zh], dtype=torch.float32, device=device)

    Zout = torch.dot(Wout, Uh)
    Y = act(Zout)

    print(Y)