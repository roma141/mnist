from __future__ import print_function, division
# import os
# from skimage import io, transform
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt

print ("starting test 2")

class mnist_load(Dataset):
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
        label = self.data_frame.ix[idx, :10].as_matrix().astype('float')
        image = self.data_frame.ix[idx, 10:].as_matrix().astype('float')
        image = image.reshape(28, 28)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

# mnist_data = mnist_load(csv_file='../train-tr.csv')

# fig = plt.figure()

# for i in range(len(mnist_data)):
#     sample = mnist_data[i]
#     # print(sample)

#     print(i, sample['image'].shape, sample['label'].shape)

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{} - {}'.format(i,np.argmax(sample['label'])))
#     ax.axis('off')
#     plt.gray()
#     plt.imshow(sample['image'])

#     if i == 3:
#         plt.show()
#         break

print("convert to tensor")

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = image.transpose((0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

mnist_data = mnist_load(csv_file='../train-tr.csv',transform=transforms.Compose([
                       ToTensor()]))

# for i in range(len(mnist_data)):
#     # print(mnist_data[i])
#     sample = mnist_data[i]

#     print(i, sample['image'].size(), sample['label'].size())

#     if i == 3:
#         break

###### before this is working

print("to dataloader")
dataloader = DataLoader(mnist_data, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    # print(i_batch)
    # print(sample_batched)
    # print(sample_batched['image'].size())
    # print(sample_batched['label'].size())

    print(i_batch, sample_batched['image'].size(),
          sample_batched['label'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        print ("batch_size:", len(sample_batched))
        # plt.figure()
        # plt.title('Batch from dataloader')
        # show_landmarks_batch(sample_batched)
        # grid = utils.make_grid(sample_batched['image'])
        # plt.imshow(grid.numpy().transpose((1, 0)))
        # for i, img in enumerate(sample_batched['image']):
        #     ax = plt.subplot(1, 4, i + 1)
        #     plt.tight_layout()
        #     ax.set_title('Sample #{}'.format(i))
        #     ax.axis('off')
        #     plt.gray()
        #     # print(img)
        #     img = img.numpy()
        #     plt.imshow(img)

        # plt.axis('off')
        # plt.ioff()
        # plt.show()
        break

other_data = datasets.MNIST('../../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))

for i in range(len(other_data)):
    # print(other_data[i])
    # print(other_data[i].size())
    sample, lbl = other_data[i]

    print(sample.size())
    print(lbl)
    plt.figure()
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    img = sample.numpy().transpose((1, 2, 0)).reshape((28, 28))
    print(img.shape)
    print(sample.numpy().shape)
    print(img.reshape((28, 28, 1)).shape)
    plt.gray()
    plt.imshow(img)
    plt.show()

    plt.gray()
    plt.imshow(img.reshape((28, 28, 1)).reshape((28, 28)))
    plt.show()

    print(i, sample.size(), lbl)

    if i == 0:
        break