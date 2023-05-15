import glob
import shutil

import pandas as pd
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
import argparse

def parse_arguments():
    parser= argparse.ArgumentParser(description='Lab3')
    parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size')
    parser.add_argument('--epochs',type=int,default=75,help='epochs')
    parser.add_argument('--gpu',action='store_true',help='use gpu')
    args=parser.parse_args()
    return args

args=parse_arguments()

# 定义超参数
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
device = 'cuda' if args.gpu else 'cpu'
test_dir = "./test"
train_dir = "./train_dataset"
val_dir = "./val_dataset"

# 设置随机种子
def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split():
    # 定义数据集路径和划分比例
    data_dir = './train'
    train_dir0 = './train_dataset'
    val_dir = './val_dataset'
    split_ratio = 0.8

    # 遍历数据集目录中的所有子目录
    for subdir in os.listdir(data_dir):
        # 构建训练集和验证集子目录的路径
        train_subdir = os.path.join(train_dir0, subdir)
        val_subdir = os.path.join(val_dir, subdir)
        # 如果不存在，创建
        os.makedirs(train_subdir, exist_ok=True)
        os.makedirs(val_subdir, exist_ok=True)
        # 获取当前目录所有文件名
        filenames = os.listdir(os.path.join(data_dir, subdir))
        # 随机打乱
        random.shuffle(filenames)
        split_idx = int(len(filenames) * split_ratio)
        # 文件拷贝
        for i, filename in enumerate(filenames):
            src_path = os.path.join(data_dir, subdir, filename)
            if i < split_idx:
                dst_path = os.path.join(train_subdir, filename)
            else:
                dst_path = os.path.join(val_subdir, filename)
            shutil.copyfile(src_path, dst_path)
# split()

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
    transforms.Resize((img_size, img_size)),
    transforms.RandomRotation(30),
    # transforms.RandomResizedCrop(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 0.485, 0.456, 0.406)，std= (0.229, 0.224, 0.225
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载训练数据
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])
])

# 加载验证数据
val_dataset = datasets.ImageFolder(val_dir, transform=train_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 加载测试数据
test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])
])
test_dataset = GetTestLoader(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



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
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=12):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




    # #初始化权重
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.constant_(m.bias, 0)




def train():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=12)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 10, 20, 40], gamma=0.1)
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
        scheduler.step()
        # 添加到tensorboard
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


def test():
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=12)
    model.load_state_dict(torch.load('./ResNetestmodel.pth'))
    model = model.to(device)
    model.eval()
    num_labels = []
    str_labels = []
    classes = datasets.ImageFolder(train_dir).class_to_idx
    dict_keys = list(classes.keys())
    dict_values = list(classes.values())
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            index, predicts = torch.max(outputs, 1)
            num_labels = num_labels + predicts.detach().cpu().numpy().tolist()
    for i in num_labels:
        index = dict_values.index(i)
        key = dict_keys[index]
        str_labels.append(key)
    file = os.listdir(test_dir)
    submission_df = pd.DataFrame({'file': file, 'species': str_labels})
    submission_df.to_csv('RESNET18.csv', index=False)

if __name__ == '__main__':
    # setseed(53)
    train()
    test()