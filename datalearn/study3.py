import ssl

from torch.utils.tensorboard import SummaryWriter

ssl._create_default_https_context = ssl._create_unverified_context

import torchvision

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./data",train=True,transform=dataset_transfrom,download=True)
test_set = torchvision.datasets.CIFAR10(root="./data",train=False,transform=dataset_transfrom,download=True)

writer = SummaryWriter("p10")

for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()