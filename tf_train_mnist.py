import tensorflow as tf
import matplotlib.pyplot as plt
from apiComon import *
import time
import scipy.io

### config
train_size = 37804
cv_size = 4196
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
hidden = 200

categories = 10
stamp = "mnist_m13_hid-"+str(hidden)+"_cs-"+str(cv_size)+"_chan-"+str(cv_channels)

print stamp
print "rounds:",rounds, "batch_size:", batch_size ,"dropout:", dropout

### queue
train_queue = tf.train.string_input_producer(["train-tr.csv"],shuffle=False)

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

label = data[:10]
img = data[10:]

data_cv = tf.decode_csv(value_cv, record_defaults=record_defaults)
y_cv = tf.stack([data_cv[:10]])
x_cv = tf.stack([data_cv[10:]])

#### train
x, y = tf.train.shuffle_batch(
    [img, label], batch_size = batch_size, 
    capacity = 10000,
min_after_dequeue = int(batch_size * 3.2))

#### cross validation
# x_cv, y_cv = tf.train.shuffle_batch(
#     [features_cv, label_cv], batch_size = batch_size, 
#     capacity = 1000,
# min_after_dequeue = 600)

#### model

W_conv1 = weight_variable([cv_size, cv_size, 1, cv_channels])
b_conv1 = bias_variable([cv_channels])
W_conv2 = weight_variable([cv_size, cv_size, cv_channels, cv_channels * 2])
b_conv2 = bias_variable([cv_channels * 2])
W_fc1 = weight_variable([img_width/4 * img_height/4 * cv_channels * 2, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)

x = tf.reshape(x, [-1, img_width, img_height,1])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_lrn1 = tf.nn.lrn(h_conv1)
h_pool1 = max_pool_2x2(h_lrn1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_lrn2 = tf.nn.lrn(h_conv2)
h_pool2 = max_pool_2x2(h_lrn2)

h_pool_flat = tf.reshape(h_pool2, [-1, img_width/4 * img_height/4  * cv_channels * 2])

h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, dropout)
pred_2 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred_2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_arg2 = tf.argmax(y, 1)

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
	# print "sleeping", sleeping
	# time.sleep(sleeping)


	loss = 0
	best_loss_cv = 10000.0
	c_epoch = 0
	for e in xrange(epochs):
		if c_epoch > 3:
			break
		print
		print "epoch", e+1
		for i in xrange(rounds):
			_, c, acc = sess.run([optimizer, cost, accuracy])

			loss += c
			print "rounds:", '%02d' % (i+1), "loss:", "{:.9f}".format(loss/(i+1)), "acc:", "{:.2f}".format(acc*100)

			#### Retrieve a single instance:
			# example, lbl = sess.run([x, y])
			# print lbl, example.shape, example[0].shape
			# img = example[0].reshape((28, 28))
			# plt.gray()
			# plt.imshow(img)
			# plt.show()
		loss = 0.0
		loss_cv = 0.0
		acc_cv = 0.0
		for i_cv in xrange(rounds_cv):
			c, acc = sess.run([cost_cv, accuracy_cv])
			loss_cv += c
			acc_cv += acc
		print 
		print "CV epoch:", '%01d' % (e+1), "loss-cv:", "{:.9f}".format(loss_cv/rounds_cv), "acc-cv:", "{:.2f}".format((acc_cv/rounds_cv)*100)
		if loss_cv/rounds_cv < best_loss_cv:
			best_loss_cv = loss_cv/rounds_cv
			c_epoch = 0
			features = {}
			features["W_conv1"] = W_conv1.eval()
			features["b_conv1"] = b_conv1.eval()
			features["W_conv2"] = W_conv2.eval()
			features["b_conv2"] = b_conv2.eval()
			# features["W_conv3"] = W_conv3.eval()
			# features["b_conv3"] = b_conv3.eval()
			# features["W_conv4"] = W_conv4.eval()
			# features["b_conv4"] = b_conv4.eval()
			# features["W_conv5"] = W_conv5.eval()
			# features["b_conv5"] = b_conv5.eval()
			# features["W_conv6"] = W_conv6.eval()
			# features["b_conv6"] = b_conv6.eval()
			# features["W_conv7"] = W_conv7.eval()
			# features["b_conv7"] = b_conv7.eval()
			# features["W_conv8"] = W_conv8.eval()
			# features["b_conv8"] = b_conv8.eval()
			features["W_fc1"] = W_fc1.eval()
			features["b_fc1"] = b_fc1.eval()
			features["W_fc2"] = W_fc2.eval()
			features["b_fc2"] = b_fc2.eval()
			name_file = stamp + "_loss-" + str(round(best_loss_cv,4))+ "_acc-" + str(round((acc_cv/rounds_cv)*100,2))
			scipy.io.savemat("temp-model/" + name_file, features, do_compression=True) 
			print "saved"
			# print
		else:
			c_epoch += 1

	coord.request_stop()
	coord.join(threads)
