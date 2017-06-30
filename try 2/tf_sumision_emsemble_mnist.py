import tensorflow as tf
import matplotlib.pyplot as plt
from apiComon import *
import time
import scipy.io

### config
file_features = "mnist_m12_hid-200_cs-5_chan-1_loss-0.0605_acc-98.05"
features1 = scipy.io.loadmat(file_features)
file_features2 = "mnist_m11_hid-200_cs-5_chan-1_loss-0.059_acc-97.69"
features2 = scipy.io.loadmat(file_features2)
file_emsembled = "mnist_emsembled4_hid-15_cs-5_chan-1_loss-0.0203_acc-99.31"
# file_emsembled = "mnist_emsembled3_hid-15_cs-5_chan-1_loss-0.0174_acc-99.4"
ems = scipy.io.loadmat(file_emsembled)
print file_features,"--", file_features2,"--",file_emsembled

train_size = 37804
cv_size = 4196
sumision_size = 0
learning_rate = 1e-2
sleeping = 5

epochs = 200
batch_size = 300

rounds = train_size / batch_size
rounds_cv = cv_size
img_width = 28
img_height = 28

dropout = 1.0
cv_size = 5
cv_channels = 1
hidden = 15
hidden2 = 200

categories = 10
stamp = "mnist_emsembled2_hid-"+str(hidden)+"_cs-"+str(cv_size)+"_chan-"+str(cv_channels)

# print stamp
# print "rounds:",rounds, "batch_size:", batch_size ,"dropout:", dropout

### queue
train_queue = tf.train.string_input_producer(["train-cv.csv"],shuffle=False)

cv_queue = tf.train.string_input_producer(["train-cv.csv"],shuffle=False)

reader = tf.TextLineReader()
_, value = reader.read(train_queue)
_, value_cv = reader.read(cv_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = []
for c in xrange(((28*28)+10)):
	record_defaults.append([0])

data = tf.decode_csv(value, record_defaults=record_defaults)
y = tf.stack([data[:10]])
x = tf.stack([data[10:]])

data_cv = tf.decode_csv(value_cv, record_defaults=record_defaults)
y_cv = tf.stack([data_cv[:10]])
x_cv = tf.stack([data_cv[10:]])

#### train

#### model
x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)

x = tf.reshape(x, [-1, img_width, img_height,1])

## model 1
W_conv1_1 = weight_variable_2([cv_size, cv_size, 1, cv_channels])
b_conv1_1 = bias_variable_2([cv_channels])
W_conv2_1 = weight_variable_2([cv_size, cv_size, cv_channels, cv_channels * 2])
b_conv2_1 = bias_variable_2([cv_channels * 2])
W_fc1_1 = weight_variable_2([img_width/4 * img_height/4 * cv_channels * 2, hidden2])
b_fc1_1 = bias_variable_2([hidden2])
W_fc2_1 = weight_variable_2([hidden2, categories])
b_fc2_1 = bias_variable_2([categories])

h_conv1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
h_lrn1_1 = tf.nn.lrn(h_conv1_1)
h_pool1_1 = max_pool_2x2(h_lrn1_1)
h_conv2_1 = tf.nn.relu(conv2d(h_pool1_1, W_conv2_1) + b_conv2_1)
h_lrn2_1 = tf.nn.lrn(h_conv2_1)
h_pool2_1 = max_pool_2x2(h_lrn2_1)

h_pool_flat_1 = tf.reshape(h_pool2_1, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1_1 = tf.nn.relu(tf.matmul(h_pool_flat_1, W_fc1_1) + b_fc1_1)
pred_1 = tf.nn.softmax(tf.matmul(h_fc1_1, W_fc2_1) + b_fc2_1)

## model 2
W_conv1_2 = weight_variable_2([cv_size, cv_size, 1, cv_channels])
b_conv1_2 = bias_variable_2([cv_channels])
W_conv2_2 = weight_variable_2([cv_size, cv_size, cv_channels, cv_channels * 2])
b_conv2_2 = bias_variable_2([cv_channels * 2])
W_fc1_2 = weight_variable_2([img_width/4 * img_height/4 * cv_channels * 2, hidden2])
b_fc1_2 = bias_variable_2([hidden2])
W_fc2_2 = weight_variable_2([hidden2, categories])
b_fc2_2 = bias_variable_2([categories])

h_conv1_2 = tf.nn.relu(conv2d(x, W_conv1_2) + b_conv1_2)
h_lrn1_2 = tf.nn.lrn(h_conv1_2)
h_pool1_2 = max_pool_2x2(h_lrn1_2)
h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
h_lrn2_2 = tf.nn.lrn(h_conv2_2)
h_pool2_2 = max_pool_2x2(h_lrn2_2)

h_pool_flat_2 = tf.reshape(h_pool2_2, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1_2 = tf.nn.relu(tf.matmul(h_pool_flat_2, W_fc1_2) + b_fc1_2)
pred_2 = tf.nn.softmax(tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2)

## model to train emsembled
fx = tf.concat([pred_1, pred_2], axis=1)
# print fx

W_fc1 = weight_variable_2([categories * 2, hidden])
b_fc1 = bias_variable_2([hidden])
W_fc2 = weight_variable_2([hidden, categories])
b_fc2 = bias_variable_2([categories])

h_fc1 = tf.nn.relu(tf.matmul(fx, W_fc1) + b_fc1)
pred2 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
pred = tf.matmul(h_fc1, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# correct_prediction = tf.equal(tf.argmax(pred2, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# y_arg2 = tf.argmax(y, 1)

### cross validation

x_cv = tf.cast(x_cv, tf.float32)
y_cv = tf.cast(y_cv, tf.float32)

x_cv = tf.reshape(x_cv, [-1, img_width, img_height,1])

## model 1
h_conv1_cv_1 = tf.nn.relu(conv2d(x_cv, W_conv1_1) + b_conv1_1)
h_lrn1_cv_1 = tf.nn.lrn(h_conv1_cv_1)
h_pool1_cv_1 = max_pool_2x2(h_lrn1_cv_1)
h_conv2_cv_1 = tf.nn.relu(conv2d(h_pool1_cv_1, W_conv2_1) + b_conv2_1)
h_lrn2_cv_1 = tf.nn.lrn(h_conv2_cv_1)
h_pool2_cv_1 = max_pool_2x2(h_lrn2_cv_1)

h_pool_flat_cv_1 = tf.reshape(h_pool2_cv_1, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1_cv_1 = tf.nn.relu(tf.matmul(h_pool_flat_cv_1, W_fc1_1) + b_fc1_1)
pred_cv_1 = tf.nn.softmax(tf.matmul(h_fc1_cv_1, W_fc2_1) + b_fc2_1)

## model 2
h_conv1_cv_2 = tf.nn.relu(conv2d(x_cv, W_conv1_2) + b_conv1_2)
h_lrn1_cv_2 = tf.nn.lrn(h_conv1_cv_2)
h_pool1_cv_2 = max_pool_2x2(h_lrn1_cv_2)
h_conv2_cv_2 = tf.nn.relu(conv2d(h_pool1_cv_2, W_conv2_2) + b_conv2_2)
h_lrn2_cv_2 = tf.nn.lrn(h_conv2_cv_2)
h_pool2_cv_2 = max_pool_2x2(h_lrn2_cv_2)

h_pool_flat_cv_2 = tf.reshape(h_pool2_cv_2, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1_cv_2 = tf.nn.relu(tf.matmul(h_pool_flat_cv_2, W_fc1_2) + b_fc1_2)
pred_cv_2 = tf.nn.softmax(tf.matmul(h_fc1_cv_2, W_fc2_2) + b_fc2_2)

## model train emsembeld
fx_cv = tf.concat([pred_cv_1, pred_cv_2], axis=1)

h_fc1_cv = tf.nn.relu(tf.matmul(fx_cv, W_fc1) + b_fc1)
pred2_cv = tf.nn.softmax(tf.matmul(h_fc1_cv, W_fc2) + b_fc2)
pred_cv = tf.matmul(h_fc1_cv, W_fc2) + b_fc2

cost_cv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_cv, labels=y_cv))
correct_prediction_cv = tf.equal(tf.argmax(pred2_cv, 1), tf.argmax(y_cv, 1))
accuracy_cv = tf.reduce_mean(tf.cast(correct_prediction_cv, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	# Start populating the filename queue.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	sess.run([
		tf.assign(W_conv1_1, features1["W_conv1"]),
		tf.assign(b_conv1_1, features1["b_conv1"][0]),
		tf.assign(W_conv2_1, features1["W_conv2"]),
		tf.assign(b_conv2_1, features1["b_conv2"][0]),
		tf.assign(W_fc1_1, features1["W_fc1"]),
		tf.assign(b_fc1_1, features1["b_fc1"][0]),
		tf.assign(W_fc2_1, features1["W_fc2"]),
		tf.assign(b_fc2_1, features1["b_fc2"][0]),
		tf.assign(W_conv1_2, features2["W_conv1"]),
		tf.assign(b_conv1_2, features2["b_conv1"][0]),
		tf.assign(W_conv2_2, features2["W_conv2"]),
		tf.assign(b_conv2_2, features2["b_conv2"][0]),
		tf.assign(W_fc1_2, features2["W_fc1"]),
		tf.assign(b_fc1_2, features2["b_fc1"][0]),
		tf.assign(W_fc2_2, features2["W_fc2"]),
		tf.assign(b_fc2_2, features2["b_fc2"][0]),
		tf.assign(W_fc1, ems["W_fc1"]),
		tf.assign(b_fc1, ems["b_fc1"][0]),
		tf.assign(W_fc2, ems["W_fc2"]),
		tf.assign(b_fc2, ems["b_fc2"][0])
		])
	# print "sleeping", sleeping
	# time.sleep(sleeping)


	loss = 0
	best_loss_cv = 10000.0
	loss = 0.0
	loss_cv = 0.0
	acc_cv = 0.0
	cheker = []
	for i_cv in xrange(rounds_cv):
		c, acc = sess.run([cost_cv, accuracy_cv])
		loss_cv += c
		acc_cv += acc
		cheker.append(i_cv)
	print 
	print "CV loss-cv:", "{:.9f}".format(loss_cv/rounds_cv), "acc-cv:", "{:.2f}".format((acc_cv/rounds_cv)*100)
	print "cheker",len(cheker)

	coord.request_stop()
	coord.join(threads)
