import numpy as np
import scipy.io

features = scipy.io.loadmat("pytorch_loss-0.1955_acc-94.47")

print features["conv1.weight"].shape
# print features["conv1.weight"]
print features["conv1.weight"][0].shape
print features["conv1.weight"][0]
# features["conv1.bias"][0]
# features["conv2.weight"]
# features["conv2.bias"][0]
# features["fc1.weight"]
# features["fc1.bias"][0]
# features["fc2.weight"]
# features["fc2.bias"][0]

### transpose
print
print "##################"
print "transpose"
print features["conv1.weight"].T.shape
print features["conv1.weight"].T
# print features["conv1.weight"].T[0].shape
# print features["conv1.weight"].T[0]
# features["conv1.bias"][0].T
# features["conv2.weight"].T
# features["conv2.bias"][0].T
# features["fc1.weight"].T
# features["fc1.bias"][0].T
# features["fc2.weight"].T
# features["fc2.bias"][0].T

name = "mnist_m1_hid-600_cs-5_chan-1_loss-0.0729_acc-97.47"
name = "resp_tfrecord_conv2_chan_8(0.0480878053862)"
features = scipy.io.loadmat(name)

print
print name
print features["W_conv1"].shape
print features["W_conv1"]
print features["W_conv1"][0].shape
print features["W_conv1"][0]

print
print "##################"
print "transpose"
print features["W_conv1"].T.shape
print features["W_conv1"].T