import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.Tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

input =torch.reshape(input,(-1,1,5,5))
# print(input.shape)

dataset = torchvision.datasets.CIFAR10("../data",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        output = self.maxpool1(input)
        return output

writer = SummaryWriter("../logs")

tudui = Tudui()
# output = tudui(input)
# print(output)

step = 0
for data in dataloader:
    imags,targets = data
    writer.add_images("input",imags,step)
    out = tudui(imags)
    writer.add_images("output",out,step)
    step = step + 1
writer.close()