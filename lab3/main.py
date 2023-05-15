import glob

import torch
import torch.nn as nn
import os
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# 定义超参数
test_dir = "./test"
train_dir = "./train"
epochs = 100
batch_size = 40
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子
def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class GetTestLoader(torch.utils.data.Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(f'{self.test_dir}/*.png'))

    def __getitem__(self, index):
        image_names = os.listdir(self.test_dir)
        image_name = os.path.join(self.test_dir, image_names[index])
        image = Image.open(image_name)
        if self.transform is not None:
            image = self.transform(image)
        return image


# 定义数据集目录和图像大小
img_size = 224
# 定义数据增强方法
# 翻转、旋转、移位等操作
train_transforms = transforms.Compose([
    # transforms.RandomRotation(30),
    transforms.Resize((img_size,img_size)),
    # transforms.RandomResizedCrop(img_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 0.485, 0.456, 0.406)，std= (0.229, 0.224, 0.225
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练数据
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试数据
test_transforms = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    # transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = GetTestLoader(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class VGG11(nn.Module):
    def __init__(self, num_classes=12):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train():
    model = VGG11()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 20, 40], gamma=0.1)
    best_loss = 999999
    tb_writer = SummaryWriter(log_dir='VGG11_sgd_lr=0.0001_bs=32_epoch=70'
                                      '/logs', comment='VGG11')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
        scheduler.step()
        # 添加到tensorboard
        train_loss = train_loss / len(train_loader.dataset)
        tb_writer.add_scalar('Train Loss', train_loss, epoch)
        print(f'Train loss: {train_loss:.4f}')
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'MODEL/' + 'VGG11_sgd_lr=0.01_bs=32_epoch=70' + '.pth')

    print(f'Best validation accuracy: {best_loss:.4f}')


if __name__ == '__main__':
    setseed(123)
    train()
