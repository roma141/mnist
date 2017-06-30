import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt


# my_data = genfromtxt('train.csv', delimiter=',')

my_data = np.loadtxt('train.csv', delimiter=',',skiprows=1,dtype="uint8")

# print my_data[0]
# line = my_data[0]
# label = line[0]
# img = line[1:]
# print img.shape
# img = img.reshape((28, 28))
# print img.shape
# print label
# plt.gray()
# plt.imshow(img)
# plt.show()

print my_data.shape
dict_labels = {}
for line in my_data:
	a = line[0]
	if str(a) in dict_labels:
		dict_labels[str(a)] += 1
	else:
		dict_labels[str(a)] = 1

print dict_labels
cv_split = 0.1
count_labels = {}
for key in dict_labels:
	count_labels[key] = 0
train = []
cv = []
print count_labels
for line in my_data:
	a = line[0]
	label = np.zeros((10,))
	label[a] = 1.0
	# print "a",a,"label",label, "label sape",label.shape, "shape line",line.shape
	# temp = line[1:]
	line = np.concatenate((label, line[1:]), axis=0)
	# print "line shape", line.shape,"test-cal", (28*28)+10
	# break
	if count_labels[str(a)] < dict_labels[str(a)] * (1.0 - cv_split):
		count_labels[str(a)] += 1
		train.append(line)
	else:
		cv.append(line)
		count_labels[str(a)] += 1

print "train:",len(train), "cv:", len(cv)
train = np.array(train)
cv = np.array(cv)
print "train:",train.shape, "cv:", cv.shape

print "saving train-tr.csv"
np.savetxt("train-tr.csv", train, fmt='%d', delimiter=',')
print "saving train-cv.csv"
np.savetxt("train-cv.csv", cv, fmt='%d', delimiter=',')

print "saved"