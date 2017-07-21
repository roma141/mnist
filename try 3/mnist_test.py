from __future__ import print_function
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# import scipy.io
import pandas as pd
import numpy as np
from api import *

# Training settings

torch.manual_seed(1)
batch = 64
test_batch = 1000
epochs = 1
log_interval = 10

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

model = Net()

temp_dict = torch.load('pytorch_loss-0.0188_acc-99.34.pt')
model.load_state_dict(temp_dict)
f_loss, f_acc = test(model, test_loader)
# name_file = "pytorch_loss-" + str(round(f_loss,4))+ "_acc-" + str(round(f_acc,2))
# torch.save(model.state_dict(), name_file +".pt")

file_name = 'train-tr.csv'
test_loader = torch.utils.data.DataLoader(
    mnist_load(file_name, transform=transforms.Compose([
                       ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

print()
print(file_name)
test(model, test_loader)

file_name = 'train-cv.csv'
test_loader = torch.utils.data.DataLoader(
    mnist_load(file_name, transform=transforms.Compose([
                       ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

print(file_name)
test(model, test_loader)


file_name = 'test.csv'
summision_loader = torch.utils.data.DataLoader(
    mnist_summ_load(file_name, transform=transforms.Compose([
                       ToTensor2(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = test_batch, shuffle=False, **kwargs)

summ = summision(model, summision_loader)

print(summ.shape)
file_summ = pd.DataFrame(data=summ)
file_summ.index += 1
file_summ.to_csv('summision_mnist_emsemble_pytorch.csv',index=True,index_label=['ImageId'],header=['Label'])
print ("end")