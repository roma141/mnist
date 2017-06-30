import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io, scipy
# import kronos
import tensorflow as tf
from PIL import Image
import csv

def fish_label(name):
    a = np.zeros(8)
    if name == "ALB":
        a[0] = 1
    elif name == "BET":
        a[1] = 1
    elif name == "DOL":
        a[2] = 1
    elif name == "LAG":
        a[3] = 1
    elif name == "NoF":
        a[4] = 1
    elif name == "OTHER":
        a[5] = 1
    elif name == "SHARK":
        a[6] = 1
    elif name == "YFT":
        a[7] = 1
        
    return a

def resize(img, high_img_w, high_img_h):
	# img.thumbnail((high_img_w, high_img_h), Image.ANTIALIAS)
	# img_w, img_h = img.size

	# background = Image.new('RGB', (high_img_w, high_img_h), (0, 0, 0))
	# bg_w, bg_h = background.size
	# offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
	# background.paste(img, offset)

	background = img.resize((high_img_w,high_img_h), Image.ANTIALIAS)	
	return background

# k = kronos.krono()
directory = "/media/roma141/Alma 2TB/torrents/mnist/original"
directory_to_save = "/media/roma141/Alma 2TB/torrents/mnist/normal"
#group = "test"
group = "train"
test_split = 0.1
test_list = {}

new_dir_train = directory_to_save + "/train-tfrecord/"
if not os.path.exists(new_dir_train):
	os.makedirs(new_dir_train)

print "deleting train..."
files1 = os.listdir(new_dir_train)
temp = len(files1)
t = 0
for file in files1:
	os.remove(new_dir_train + file)
	print "del train", t*100.0/temp
	t += 1

new_dir_test = directory_to_save + "/test-tfrecord/"
if not os.path.exists(new_dir_test):
	os.makedirs(new_dir_test)

print "deleting test..."
files1 = os.listdir(new_dir_test)
temp = len(files1)
t = 0
for file in files1:
	os.remove(new_dir_test + file)
	print "del test", t*100.0/temp
	t += 1

total_count = 0
count_by_label = {}
all_train = {}
with open(directory + '/train.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		if total_count == 0:
			total_count += 1
			continue
		if str(row[0]) in count_by_label:
			count_by_label[str(row[0])] += 1
		else:
			count_by_label[str(row[0])] = 1

		if str(row[0]) in all_train:
			all_train[str(row[0])].append([row[1:]])
		else:
			all_train[str(row[0])] = [row[1:]]
		# print ', '.join(row)
		# print row[0]
		# print row[1:]
		# temp = np.array(row[1:],dtype=np.float32)
		# print temp.shape
		# temp = temp.reshape(28,28)
		# print temp.shape
		# plt.gray()
		# plt.imshow(temp)
		# plt.show()
		total_count += 1

# print count_by_label
# print len(all_train["1"])
for key in count_by_label:
	to_test = int(count_by_label[key] * test_split)
	test_list[key] = to_test

# print test_list
total_count2 = 0
train_count = 0
test_count = 0

for key in count_by_label:
	count = 0
	label = np.zeros(10)
	label[int(key)] = 1
	for temp in all_train[key]:
		if count >= count_by_label[key] - test_list[key]:
			new_dir = new_dir_test
			if_test = 1
		else:
			new_dir = new_dir_train
			if_test = 0
		# print key
		print key, round(total_count2*100.0/total_count,2)
		total_count2 += 1

		image = np.array(temp,dtype=np.float32)
		# print image.shape
		image = image.reshape(28,28)
		# print image.shape
		# plt.gray()
		# plt.imshow(image)
		# plt.show()
		file_name = key + "_" + str(count)
		writer = tf.python_io.TFRecordWriter(new_dir + "/" + file_name + ".tfrecords")
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'label': tf.train.Feature(
						float_list=tf.train.FloatList(value=label)),
					'image': tf.train.Feature(
						float_list=tf.train.FloatList(value=image.flatten().tolist())),
			}))

		writer.write(example.SerializeToString())
		writer.close()
		if count >= count_by_label[key] - test_list[key]:
			test_count += 1
		else:
			train_count += 1
		count += 1

print "total_count:",total_count, "train count:",train_count,"test count:",test_count
print "test dict:",test_list
print "end"