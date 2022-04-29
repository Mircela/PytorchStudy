import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])

input = torch.reshape(input,(-1,1,2,2))
# print(input.shape)
dataset = torchvision.datasets.CIFAR10("../data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sig1 = Sigmoid()

    def forward(self,x):
        out = self.sig1(x)
        return out

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs,tags = data
    writer.add_images("in",imgs,global_step=step)
    out = tudui(imgs)
    writer.add_images("out",out,global_step=step)
    step = step + 1

writer.close()