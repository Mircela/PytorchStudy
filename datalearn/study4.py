import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

test_loader = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

img,tag = test_data[1]
# print(img.shape)
# print(tag)

writer = SummaryWriter("data_loader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,tags = data
        writer.add_images("epoch:{}".format(epoch),imgs,step)
        step = step + 1
writer.close()