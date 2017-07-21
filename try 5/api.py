from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=5)
        self.conv2 = nn.Conv2d(1, 2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.size())
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.size())
        x = x.view(-1, 32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
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
        return image.float().div(255), label


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
        return image.float().div(255)

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
        # data2 = torch.cat([output1, output2])
        data2 = [output1,output2]
        data2 = torch.cat(data2, 1)
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

try:
    import accimage
except ImportError:
    accimage = None


class ToTensor_type2(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            print("## 1")
            pic = np.where(pic < 128, pic, 0)
            pic = np.where(pic > 127, pic, 255) 
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            print("## 2")
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            pic = np.where(pic < 128, pic, 0)
            pic = np.where(pic > 127, pic, 255) 
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            print("-- 1")
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            print("-- 2")
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            # print("-- 3")
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.numpy()
            # img = np.where(img <= 127, img, 255) # 2
            # img = np.where(img > 127, img, 0) # 2
            img = np.where(img <= 170, img, 255) # 3
            # img = np.where((img < 85) & (img > 170), img, 127) # 3
            img = np.where(img > 85, img, 0) # 3
            img = torch.from_numpy(img)
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            # print("## 3")
            return img.float().div(255)
        else:
            print("## 4")
            return img