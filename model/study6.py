import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch import nn

dataset = torchvision.datasets.CIFAR10("../data",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64,drop_last=True)

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(196608,10)

    def forward(self,x):
        out = self.linear1(x)
        return out

tudui = Tudui()

for data in dataloader:
    imgs,tags = data
    print(imgs.shape)
    out = torch.flatten(imgs)
    print(out.shape)
    out = tudui(out)
    print(out.shape)
