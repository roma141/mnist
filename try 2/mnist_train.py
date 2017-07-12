from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.io
import pandas as pd
import numpy as np

# Training settings

# torch.manual_seed(1)
batch = 64
test_batch = 1000
epochs = 100
log_interval = 1

kwargs = {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size = batch, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

print("network")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

model = Net()
print(model)
# print(model.state_dict())
print(model.state_dict().keys())

print("optimizer")
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
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

def test(epoch):
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

print("training...")
best_low_loss = 100000
conti = 0
for epoch in range(1, epochs + 1):
    train(epoch)
    f_loss, f_acc = test(epoch)
    if f_loss < best_low_loss:
        best_low_loss = f_loss
        conti = 0
        temp_dict = model.state_dict()
        # name_file = "pytorch_loss-" + str(round(f_loss,4))+ "_acc-" + str(round(f_acc,2))
        # torch.save(model.state_dict(), "pytorch/" + name_file +".dat")
        continue
    else:
        if conti < 5:
            conti += 1
        else:
            break
model.load_state_dict(temp_dict)
f_loss, f_acc = test(epoch)

# print(model.state_dict())

# print("saving mat file")
# name_file = "pytorch_loss-" + str(round(f_loss,4))+ "_acc-" + str(round(f_acc,2))
# data = {}
# # f_data = model.state_dict()
# # print (data)
# for key in model.state_dict().keys():
#     data[key] = model.state_dict()[key].numpy()
# # print (data)
# scipy.io.savemat("pytorch/" + name_file, data, do_compression=True)

# torch.save("pytorch/" + name_file, model.state_dict() [, format, referenced])
# torch.save(model.state_dict(), "pytorch/" + name_file +".dat")
# print ("starting test 2")

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
        image = torch.from_numpy(image)
        # return image.type(torch.FloatTensor), label
        return image, label

file_name = 'train-tr.csv'
test_loader = torch.utils.data.DataLoader(
    mnist_load(file_name, transform=transforms.Compose([
                       ToTensor(),
                       # transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

print()
print(file_name)
test(epoch)

file_name = 'train-cv.csv'
test_loader = torch.utils.data.DataLoader(
    mnist_load(file_name, transform=transforms.Compose([
                       ToTensor(),
                       # transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

print(file_name)
test(epoch)

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

class ToTensor(object):
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

file_name = 'test.csv'
summision_loader = torch.utils.data.DataLoader(
    mnist_summ_load(file_name, transform=transforms.Compose([
                       ToTensor(),
                       # transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=False, **kwargs)

def summision():
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


summ = summision()

print(summ.shape)
file_summ = pd.DataFrame(data=summ)
# file_summ.columns = ['ImageId','Label']
file_summ.index += 1
# print(file_summ)
file_summ.to_csv('summision_mnist_pytorch.csv',index=True,index_label=['ImageId'],header=['Label'])
print ("end")