from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.io

# Training settings

torch.manual_seed(1)
batch = 64
test_batch = 1000
epochs = 1
log_interval = 1

kwargs = {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                       transforms.Normalize((0.5,), (0.5,))
                   ])),
    batch_size = batch, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                       transforms.Normalize((0.5,), (0.5,))
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
for epoch in range(1, epochs + 1):
    train(epoch)
f_loss, f_acc = test(epoch)

# print(model.state_dict())

print("saving mat file")
name_file = "pytorch_loss-" + str(round(f_loss,4))+ "_acc-" + str(round(f_acc,2))
data = {}
# f_data = model.state_dict()
# print (data)
for key in model.state_dict().keys():
    data[key] = model.state_dict()[key].numpy()
# print (data)
scipy.io.savemat("pytorch/" + name_file, data, do_compression=True)

# torch.save("pytorch/" + name_file, model.state_dict() [, format, referenced])
# torch.save(model.state_dict(), "pytorch/" + name_file +".dat")
print ("end")