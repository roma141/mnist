from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np


class net_conv(nn.Module):
    def __init__(self):
        super(net_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.size())
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
        # return x

class net_mlp(nn.Module):
    def __init__(self):
        super(net_mlp, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
        # return x

class net_total(nn.Module):
    def __init__(self):
        super(net_total, self).__init__()
        self.net1 = net_conv()
        # self.net2 = net_mlp()
        self.net2 = net_conv()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 10)
        # self.net1 = net_conv()
        # self.net2 = net_mlp()

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        # x = torch.cat([x1,x2])
        x = [x1,x2]
        x = torch.cat(x, 1)
        # x = F.dropout(x, training=self.training)
        x = x.view(-1, 20)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class mnist_load(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        label = self.data_frame.ix[idx, :10].as_matrix().astype('float32')
        label = np.argmax(label)
        image = self.data_frame.ix[idx, 10:].as_matrix().astype('float32')
        image = image.reshape((28,28,1)).transpose((2, 0, 1))
        sample = [image, label]

        if self.transform:
            image, label = self.transform(sample)

        return image, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.reshape((28,28,1)).transpose((2, 0, 1))
        # image = image.transpose((0, 1))
        # print(image)
        image = torch.from_numpy(image)
        # return image.type(torch.FloatTensor), label
        return image, label

class mnist_summ_load(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.data_frame.ix[idx, :].as_matrix().astype('float32')
        image = image.reshape((28,28,1)).transpose((2, 0, 1))
        sample = image

        if self.transform:
            image = self.transform(sample)

        return image

class ToTensor2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.reshape((28,28,1)).transpose((2, 0, 1))
        # image = image.transpose((0, 1))
        image = torch.from_numpy(image)
        return image

def train(epoch, model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # print(data.size(), target.size())
        # print(data.type(), target.type())
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss, 100. * correct / len(test_loader.dataset))

def train2(epoch, model, train_loader, optimizer, log_interval, model1, model2):
    model.train()
    # model1.eval()
    # model2.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        output1 = model1(data)
        output2 = model2(data)
        # print(model1(data).requires_grad)
        optimizer.zero_grad()
        data2 = torch.cat([output1, output2])
        # print(model(data2).requires_grad)
        output = model(data2)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test2(model, test_loader, model1, model2):
    model.eval()
    # model1.eval()
    # model2.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # print(data.size(), target.size())
        # print(data.type(), target.type())
        data, target = Variable(data, volatile=True), Variable(target)
        output1 = model1(data)
        output2 = model2(data)
        data2 = torch.cat([output1, output2])
        output = model(data2)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (test_loss, 100. * correct / len(test_loader.dataset))

def summision(model, summision_loader):
    model.eval()
    for data in summision_loader:
        # print(data.size(), target.size())
        # print(data.type(), target.type())
        data = Variable(data, volatile=True)
        output = model(data)
        # print(output)
        pred = output.data.max(1)[1]
        if 'summision' in locals():
            summision = np.concatenate((summision,pred.numpy()))
            # print("concatenate more")
        else:
            summision = pred.numpy()
    return summision