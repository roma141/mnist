import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # print("ok")
        self.conv1 = nn.Conv2d(1, 20, 5)
        print("ok")
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))


net = Model()
print("1")
print(net)
print("2")

# test for github