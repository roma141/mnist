from __future__ import print_function
import torch
# import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
# import scipy.io
import pandas as pd
import numpy as np
import copy
from api import *
import matplotlib.pyplot as plt


batch = 4
test_batch = 1000
epochs = 1
log_interval = 10

kwargs = {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       # transforms.ToTensor(),
                       ToTensor_type2(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size = batch, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       # transforms.ToTensor(),
                       ToTensor_type2(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                       # transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size = test_batch, shuffle=True, **kwargs)

for i_batch, sample_batched in enumerate(train_loader):
    # print(i_batch)
    # print(sample_batched)
    # print(sample_batched['image'].size())
    # print(sample_batched['label'].size())

    print(i_batch, sample_batched[0].size(),
          sample_batched[1].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        print ("batch_size:", len(sample_batched))
        plt.figure()
        plt.title('Batch from dataloader')
        # show_landmarks_batch(sample_batched)
        grid = utils.make_grid(sample_batched[0])
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        # plt.imshow(grid.numpy())
        for i, img in enumerate(sample_batched[0]):
            print(img)
            break
        #     ax = plt.subplot(1, 4, i + 1)
        #     plt.tight_layout()
        #     ax.set_title('Sample #{}'.format(i))
        #     ax.axis('off')
        #     # plt.gray()
        #     # print(img)
        #     img = img.numpy().transpose((1, 2, 0))
        #     # img = img.numpy()
        #     print(img.shape)
        #     plt.imshow(img)

        plt.axis('off')
        plt.ioff()
        plt.show()
        break