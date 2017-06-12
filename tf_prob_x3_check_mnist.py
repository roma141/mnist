import tensorflow as tf
import matplotlib.pyplot as plt
from apiComon import *
import time
import scipy.io
import numpy as np

### config
file_features = "mnist_m1_loss-1.34"
features = scipy.io.loadmat(file_features)
file_features2 = "mnist_m2_loss-0.94"
features2 = scipy.io.loadmat(file_features2)
file_features3 = "mnist_m4_hid-25_cs-5_chan-1_loss-0.377_acc-89.56"
features3 = scipy.io.loadmat(file_features3)
print file_features,"--", file_features2,"--", file_features3

train_size = 37804
cv_size = 4196
learning_rate = 1e-4
sleeping = 5

epochs = 90
batch_size = 500

rounds = train_size / batch_size
rounds_cv = cv_size
img_width = 28
img_height = 28

dropout = 0.5
cv_size = 5
cv_channels = 1
hidden = 25

categories = 10

print "rounds:",rounds, "batch_size:", batch_size ,"dropout:", dropout

### queue
cv_queue = tf.train.string_input_producer(["train-cv.csv"])

reader = tf.TextLineReader()
_, value_cv = reader.read(cv_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = []
for c in xrange(((28*28)+10)):
	record_defaults.append([0])


data_cv = tf.decode_csv(value_cv, record_defaults=record_defaults)
y_cv = tf.stack([data_cv[:10]])
x_cv = tf.stack([data_cv[10:]])


#### model

W_conv1 = weight_variable([cv_size, cv_size, 1, cv_channels])
b_conv1 = bias_variable([cv_channels])
W_conv2 = weight_variable([cv_size, cv_size, cv_channels, cv_channels * 2])
b_conv2 = bias_variable([cv_channels * 2])
W_fc1 = weight_variable([img_width/4 * img_height/4 * cv_channels * 2, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

### cross validation

x_cv = tf.cast(x_cv, tf.float32)
y_cv = tf.cast(y_cv, tf.float32)

x_cv = tf.reshape(x_cv, [-1, img_width, img_height,1])

h_conv1_cv = tf.nn.relu(conv2d(x_cv, W_conv1) + b_conv1)
h_lrn1_cv = tf.nn.lrn(h_conv1_cv)
h_pool1_cv = max_pool_2x2(h_lrn1_cv)
h_conv2_cv = tf.nn.relu(conv2d(h_pool1_cv, W_conv2) + b_conv2)
h_lrn2_cv = tf.nn.lrn(h_conv2_cv)
h_pool2_cv = max_pool_2x2(h_lrn2_cv)

h_pool_flat_cv = tf.reshape(h_pool2_cv, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1_cv = tf.nn.relu(tf.matmul(h_pool_flat_cv, W_fc1) + b_fc1)
pred_cv_2 = tf.nn.softmax(tf.matmul(h_fc1_cv, W_fc2) + b_fc2)
pred_cv = tf.matmul(h_fc1_cv, W_fc2) + b_fc2

cost_cv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_cv, labels=y_cv))
correct_prediction_cv = tf.equal(tf.argmax(pred_cv_2, 1), tf.argmax(y_cv, 1))
accuracy_cv = tf.reduce_mean(tf.cast(correct_prediction_cv, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	# Start populating the filename queue.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	sess.run([
		tf.assign(W_conv1,features["W_conv1"]),
		tf.assign(b_conv1,features["b_conv1"][0]),
		tf.assign(W_conv2,features["W_conv2"]),
		tf.assign(b_conv2,features["b_conv2"][0]),
		tf.assign(W_fc1,features["W_fc1"]),
		tf.assign(b_fc1,features["b_fc1"][0]),
		tf.assign(W_fc2,features["W_fc2"]),
		tf.assign(b_fc2,features["b_fc2"][0]),
		])

	loss = 0
	best_loss_cv = 10000.0
	c_epoch = 0
	loss = 0.0
	loss_cv = 0.0
	acc_cv = 0.0
	prob1 = []
	for i_cv in xrange(rounds_cv):
		c, acc,p = sess.run([cost_cv, accuracy_cv, pred_cv_2])
		loss_cv += c
		acc_cv += acc
		prob1.append(p[0])
	print 
	print "CV ", "loss-cv:", "{:.9f}".format(loss_cv/rounds_cv), "acc-cv:", "{:.2f}".format((acc_cv/rounds_cv)*100)

	sess.run([
		tf.assign(W_conv1,features2["W_conv1"]),
		tf.assign(b_conv1,features2["b_conv1"][0]),
		tf.assign(W_conv2,features2["W_conv2"]),
		tf.assign(b_conv2,features2["b_conv2"][0]),
		tf.assign(W_fc1,features2["W_fc1"]),
		tf.assign(b_fc1,features2["b_fc1"][0]),
		tf.assign(W_fc2,features2["W_fc2"]),
		tf.assign(b_fc2,features2["b_fc2"][0]),
		])

	loss = 0
	best_loss_cv = 10000.0
	c_epoch = 0
	loss = 0.0
	loss_cv = 0.0
	acc_cv = 0.0
	prob2 = []
	for i_cv in xrange(rounds_cv):
		c, acc,p = sess.run([cost_cv, accuracy_cv, pred_cv_2])
		loss_cv += c
		acc_cv += acc
		prob2.append(p[0])
	print 
	print "CV ", "loss-cv:", "{:.9f}".format(loss_cv/rounds_cv), "acc-cv:", "{:.2f}".format((acc_cv/rounds_cv)*100)

	sess.run([
		tf.assign(W_conv1,features3["W_conv1"]),
		tf.assign(b_conv1,features3["b_conv1"][0]),
		tf.assign(W_conv2,features3["W_conv2"]),
		tf.assign(b_conv2,features3["b_conv2"][0]),
		tf.assign(W_fc1,features3["W_fc1"]),
		tf.assign(b_fc1,features3["b_fc1"][0]),
		tf.assign(W_fc2,features3["W_fc2"]),
		tf.assign(b_fc2,features3["b_fc2"][0]),
		])

	loss = 0
	best_loss_cv = 10000.0
	c_epoch = 0
	loss = 0.0
	loss_cv = 0.0
	acc_cv = 0.0
	prob3 = []
	for i_cv in xrange(rounds_cv):
		c, acc,p = sess.run([cost_cv, accuracy_cv, pred_cv_2])
		loss_cv += c
		acc_cv += acc
		prob3.append(p[0])
	print 
	print "CV ", "loss-cv:", "{:.9f}".format(loss_cv/rounds_cv), "acc-cv:", "{:.2f}".format((acc_cv/rounds_cv)*100)


	prob1 = np.array(prob1)
	prob2 = np.array(prob2)
	prob3 = np.array(prob3)

	print prob1.shape, prob2.shape, prob3.shape

	best1 = prob1 - prob2
	best2 = prob1 - prob3
	best3 = prob2 - prob3

	print
	print "best1 shape:" , best1.shape
	print "1:", np.sum(best1)
	print "2:", np.sum(np.absolute(best1))

	print
	print "best2 shape:" , best2.shape
	print "1:", np.sum(best2)
	print "2:", np.sum(np.absolute(best2))

	print
	print "best3 shape:" , best3.shape
	print "1:", np.sum(best3)
	print "2:", np.sum(np.absolute(best3))

	coord.request_stop()
	coord.join(threads)
