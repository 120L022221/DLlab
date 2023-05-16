import math
import glob
import time
import pandas as pd
import torch
import torch.nn as nn
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Lab3')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=75, help='epochs')
    parser.add_argument('--gpu', action='store_true', help='use gpu')
    args = parser.parse_args()
    return args


args = parse_arguments()
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
# device = 'cuda' if args.gpu else 'cpu'
device = 'cuda'
test_dir = "./test"
train_dir = "./train_dataset"
val_dir = "./val_dataset"
img_size = 224


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=12, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

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


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=12, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


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


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_dataset = GetTestLoader(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def adjust_learning_rate(optimizer,epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    model=eca_resnet18(k_size=[3, 3, 3, 3], num_classes=12, pretrained=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 5, 10, 20, 40], gamma=0.1)
    best_valid_acc = 0.0
    tb_writer = SummaryWriter(log_dir='resnet_eca_sgd_lr=0.0001_bs=32_epoch=75'
                                      '/logs', comment='resnet18_eca')
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
        adjust_learning_rate(optimizer, epoch)
        # scheduler.step()
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
            torch.save(model.state_dict(), 'MODEL/' + 'resnet_eca_sgd_lr=0.0001_bs=32_epoch=75' + '.pth')

    print(f'Best validation accuracy: {best_valid_acc:.4f}')


def test():
    # 测试所用时间
    model = eca_resnet18(k_size=[3, 3, 3, 3], num_classes=12, pretrained=False)
    model.load_state_dict(torch.load('MODEL/' + 'resnet_eca_sgd_lr=0.0001_bs=32_epoch=75' + '.pth'))
    model = model.to(device)
    model.eval()
    num_labels = []
    str_labels = []
    classes = datasets.ImageFolder(train_dir).class_to_idx
    dict_keys = list(classes.keys())
    dict_values = list(classes.values())
    start_time = time.time()
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            index, predicts = torch.max(outputs, 1)
            num_labels = num_labels + predicts.detach().cpu().numpy().tolist()
    # 测试时间
    end_time = time.time()
    print('Image/s', 794.0/(end_time - start_time))
    for i in num_labels:
        index = dict_values.index(i)
        key = dict_keys[index]
        str_labels.append(key)
    file = os.listdir(test_dir)
    submission_df = pd.DataFrame({'file': file, 'species': str_labels})
    submission_df.to_csv('resnet_eca_sgd_0.0001.csv', index=False)


if __name__ == '__main__':
    train()
    test()