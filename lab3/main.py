import torch
import torch.nn as nn
import os
import shutil
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

# 定义超参数
test_dir="./test"
epochs=70
batch_size = 32
learning_rate = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize_images(dataset_path):
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".png"):
                image = Image.open(file_path)
                image = image.resize((224, 224), resample=Image.ANTIALIAS)
                image.save(file_path)

# resize_images("./train")
# resize_images("./test")

# 设置随机种子
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
#
# # 定义数据集路径和划分比例
# data_dir = './train'
train_dir0 = './train_dataset'
val_dir = './val_dataset'
# split_ratio = 0.9
#
# # 遍历数据集目录中的所有子目录
# for subdir in os.listdir(data_dir):
#     # 构建训练集和验证集子目录的路径
#     train_subdir = os.path.join(train_dir0, subdir)
#     val_subdir = os.path.join(val_dir, subdir)
#
#     # 如果训练集和验证集子目录不存在，则创建它们
#     os.makedirs(train_subdir, exist_ok=True)
#     os.makedirs(val_subdir, exist_ok=True)
#
#     # 获取当前子目录中的所有文件名
#     filenames = os.listdir(os.path.join(data_dir, subdir))
#
#     # 随机打乱文件名列表
#     random.shuffle(filenames)
#
#     # 计算划分点
#     split_idx = int(len(filenames) * split_ratio)
#
#     # 将文件拷贝到训练集和验证集目录中
#     for i, filename in enumerate(filenames):
#         src_path = os.path.join(data_dir, subdir, filename)
#
#         if i < split_idx:
#             dst_path = os.path.join(train_subdir, filename)
#         else:
#             dst_path = os.path.join(val_subdir, filename)
#
#         shutil.copyfile(src_path, dst_path)



def getStat(train_data=datasets.ImageFolder(train_dir0,transform=transforms.ToTensor()),
            val_data=datasets.ImageFolder(val_dir,transform=transforms.ToTensor())):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :param test_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader1 = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    train_mean = torch.zeros(3)
    train_std = torch.zeros(3)
    for inputs,labels in train_loader1:
        for d in range(3):
            train_mean[d] += inputs[:, d, :, :].mean()
            train_std[d] += inputs[:, d, :, :].std()
    train_mean.div_(len(train_data))
    train_std.div_(len(train_data))

    print('Compute mean and variance for val data.')
    print(len(val_data))
    test_loader1 = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    test_mean = torch.zeros(3)
    test_std = torch.zeros(3)
    for inputs, labels in test_loader1:
        for d in range(3):
            test_mean[d] += inputs[:, d, :, :].mean()
            test_std[d] += inputs[:, d, :, :].std()
    test_mean.div_(len(val_data))
    test_std.div_(len(val_data))

    return list(train_mean.numpy()), list(train_std.numpy()),list(test_mean.numpy()),list(test_std.numpy())
#
# print(getStat())

# 定义数据集目录和图像大小
img_size = 224

# 定义数据增强方法
# 翻转、旋转、移位等操作
train_transforms = transforms.Compose([
    # transforms.RandomRotation(30),
    # transforms.Resize(img_size),
    # transforms.RandomResizedCrop(img_size),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 0.485, 0.456, 0.406)，std= (0.229, 0.224, 0.225
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载训练数据
train_dataset = datasets.ImageFolder(train_dir0, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 加载测试数据
val_transforms = transforms.Compose([
    # transforms.Resize(img_size),
    # transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)


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
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train():
    model=VGG11()
    model = model.to(device)
    criterion=nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler =torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2,5,10,20],gamma=0.1)
    best_valid_acc = 0.0
    tb_writer = SummaryWriter(log_dir='VGG11_sgd_lr=0.0001_bs=32_epoch=70'
                                      '/logs', comment='VGG11')
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        model.train()
        train_loss = 0.0
        train_acc = 0.0
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
            train_acc += torch.sum(preds == labels.data)
        #添加到tensorboard
        train_loss = train_loss / len(train_loader.dataset)
        tb_writer.add_scalar('Train Loss', train_loss, epoch)
        train_acc = train_acc / len(train_loader.dataset)
        tb_writer.add_scalar('Train Acc', train_acc, epoch)
        print(f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}')

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        for inputs, labels in val_loader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                valid_acc += torch.sum(preds == labels.data)
        valid_loss = valid_loss / len(val_loader.dataset)
        valid_acc = valid_acc / len(val_loader.dataset)
        tb_writer.add_scalar('Valid Loss', valid_loss, epoch)
        tb_writer.add_scalar('Valid Acc', valid_acc, epoch)
        print(f'Valid loss: {valid_loss:.4f}, Valid acc: {valid_acc:.4f}')
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'MODEL/' + 'VGG11_sgd_lr=0.0001_bs=32_epoch=70' + '.pth')

    print(f'Best validation accuracy: {best_valid_acc:.4f}')

if __name__ == '__main__':
    train()