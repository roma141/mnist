import tensorflow as tf
import numpy as np
from apiDeepLearning import *
import scipy.io
from params import param
import time
import os
import csv
import matplotlib.pyplot as plt

features = scipy.io.loadmat("resp_tfrecord_conv2_chan_8")
# features = scipy.io.loadmat("resp_tfrecord_conv2_chan_8(0.0505910888096)")
sub_file = "submission_7.csv"
parameters = param()
directory = "/media/roma141/Alma 2TB/torrents/mnist/original"
samples = 0
# count_by_label = {}
# all_train = {}
with open(directory + '/test.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		if samples == 0:
			samples += 1
			continue
		# if str(row[0]) in count_by_label:
		# 	count_by_label[str(row[0])] += 1
		# else:
		# 	count_by_label[str(row[0])] = 1

		# if str(row[0]) in all_train:
		# 	all_train[str(row[0])].append([row[1:]])
		# else:
		# 	all_train[str(row[0])] = [row[1:]]
# path = "../../data/fish/sumission-tfrecord/"
# files = [path + file for file in os.listdir(path)]
# samples = len(files)

hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
cv_all_size = 5
cv_all_channels = 8
last_img_size = 7
channels_jpg = 1

# filename_queue = tf.train.string_input_producer(files, shuffle=False)
# reader = tf.TFRecordReader()
# tfname, data = reader.read(filename_queue)
    
# example = tf.parse_single_example(
#     data,
#     features = {
#         'image': tf.FixedLenFeature([224, 224, 3], tf.float32),
#     }
# )

# x = example["image"]
x = tf.placeholder(tf.float32, shape=[None, img_width, img_height])

W_conv1 = weight_variable([cv_all_size, cv_all_size, channels_jpg, cv_all_channels])
b_conv1 = bias_variable([cv_all_channels])
W_conv2 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels * 2])
b_conv2 = bias_variable([cv_all_channels * 2])
# W_conv3 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 4])
# b_conv3 = bias_variable([cv_all_channels * 4])
# W_conv4 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 8])
# b_conv4 = bias_variable([cv_all_channels * 8])
# W_conv5 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 16])
# b_conv5 = bias_variable([cv_all_channels * 16])
# W_conv6 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 16, cv_all_channels * 32])
# b_conv6 = bias_variable([cv_all_channels * 32])
# W_conv7 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 32, cv_all_channels * 64])
# b_conv7 = bias_variable([cv_all_channels * 64])
# W_conv8 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv8 = bias_variable([cv_all_channels])
# W_conv9 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv9 = bias_variable([cv_all_channels])
# W_conv10 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv10 = bias_variable([cv_all_channels])

W_fc1 = weight_variable([last_img_size * last_img_size * cv_all_channels * 2, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x_r = tf.reshape(tf.cast(x, tf.float32), [-1,img_width,img_height,channels_jpg])

# conv
h_conv1 = tf.nn.relu(conv2d(x_r, W_conv1) + b_conv1)
h_lrn1 = conv2d_lrn(h_conv1)
h_pool1 = max_pool_2x2(h_lrn1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_lrn2 = conv2d_lrn(h_conv2)
h_pool2 = max_pool_2x2(h_lrn2)
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)
# h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
# h_pool5 = max_pool_2x2(h_conv5)
# h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
# h_pool6 = max_pool_2x2(h_conv6)
# h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
# h_pool7 = max_pool_2x2(h_conv7)
# h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
# h_pool8 = max_pool_2x2(h_conv8)
# h_conv9 = tf.nn.relu(conv2d(h_pool8, W_conv9) + b_conv9)
# h_pool9 = max_pool_2x2(h_conv9)
# h_conv10 = tf.nn.relu(conv2d(h_pool9, W_conv10) + b_conv10)
# h_pool10 = max_pool_2x2(h_conv10)

h_pool3 = aver_pool_7x7(h_pool2, 1)
# h_pool3 = max_pool_7x7(aver_pool_7x7(h_pool2, 1), 1)
h_pool_last_flat = tf.reshape(h_pool3, [-1, last_img_size * last_img_size  * cv_all_channels * 2])

h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, 1)
pred = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
print pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
pred_arg2 = tf.argmax(pred, 1)
# y_arg2 = tf.argmax(y, 1)
# print "y",y
# print "pred", pred

init = tf.global_variables_initializer()
acc_total = 0.0

with tf.Session() as sess:
	sess.run(init)
	sess.run([
	    W_conv1.assign(features["W_conv1"]),
	    b_conv1.assign(features["b_conv1"][0]),
	    W_conv2.assign(features["W_conv2"]),
	    b_conv2.assign(features["b_conv2"][0]),
	    # W_conv3_2.assign(features["W_conv3"]),
	    # b_conv3_2.assign(features["b_conv3"][0]),
	    # W_conv4_2.assign(features["W_conv4"]),
	    # b_conv4_2.assign(features["b_conv4"][0]),
	    # W_conv5_2.assign(features["W_conv5"]),
	    # b_conv5_2.assign(features["b_conv5"][0]),
	    W_fc1.assign(features["W_fc1"]),
	    b_fc1.assign(features["b_fc1"][0]),
	    W_fc2.assign(features["W_fc2"]),
	    b_fc2.assign(features["b_fc2"][0])
    ])
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	r = []
	r.append(["ImageId","Label"])
	step = 0
	with open(directory + '/test.csv', 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		for row in spamreader:
			print "step:",step
			if step == 0:
				step += 1
				continue
			# print len(row), row[783]
			image = np.array(row, dtype=np.float32)
			# print image.shape
			image = image.reshape(28,28)
			# plt.gray()
			# plt.imshow(image)
			# plt.show()
			image = image.reshape(1,28,28)

			prob,pred_arg = sess.run([pred,pred_arg2],{ 
				x: image                        
			  })
			prob = prob[0]
			r.append([step, pred_arg[0]])
			step += 1
			# if step > 5:
			# 	break
	csvfile.close()
	print "ImageId, Label"
	np.savetxt(sub_file, r, delimiter=',', fmt="%s,%s")


	coord.request_stop()
	try:
		coord.join(threads)
	except:
		pass
	sess.close()
print sub_file