import torch
import torchvision

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练集的长度：{}".format(train_data_size))
print("测试集的长度：{}".format(test_data_size))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

device = torch.device("cuda")
print(device)


class Tudui(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.module(x)
        return x


tudui = Tudui()
tudui = tudui.to(device)
# if torch.cuda.is_available():
#     tudui = tudui.cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()
loss_fn = loss_fn.to(device)
# 优化器
# learn_rate = 0.01
learn_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learn_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs")
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))
    tudui.train()
    start = time.time()
    for data in train_dataloader:
        imgs, tags = data
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     tags = tags.cuda()
        imgs = imgs.to(device)
        tags = tags.to(device)
        out = tudui(imgs)
        loss = loss_fn(out, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end = time.time()
            print(end - start)
            print("-----训练次数：{}，Loss={}-----".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, tags = data
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     tags = tags.cuda()
            imgs = imgs.to(device)
            tags = tags.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, tags)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == tags).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的loss={}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_test_accuracy / test_data_size))

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_test_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "./save/train_step{}.pth".format(i))
    print("模型已保存")

writer.close()
