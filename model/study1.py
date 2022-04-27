import torch
from torch import nn


class MyFirstModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        output = x + 1
        return output

mfm = MyFirstModel()
x = torch.tensor(1.0)
out = mfm(x)
print(out)

