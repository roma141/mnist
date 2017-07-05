import tensorflow as tf
import numpy as np
from apiDeepLearning import *
import scipy.io
from params import param
import time
import os

# features = scipy.io.loadmat("mnist_m1_hid-600_cs-5_chan-1_loss-0.0729_acc-97.47")
# features = scipy.io.loadmat("mnist_m1_hid-50_cs-5_chan-1_loss-0.1363_acc-95.59")
features = scipy.io.loadmat("mnist_m1_t2_hid-600_cs-5_chan-1_loss-0.2322_acc-92.3")
parameters = param()
directory = "/media/roma141/Alma 2TB/torrents/mnist/normal"
path = directory + "/test-tfrecord/"
files = [path + file for file in os.listdir(path)]
samples = len(files)

hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
cv_all_size = 5
cv_all_channels = 1
last_img_size = 7
channels_jpg = 1

filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
tfname, data = reader.read(filename_queue)
    
example = tf.parse_single_example(
    data,
    features = {
        'label': tf.FixedLenFeature([10], tf.float32),
        'image': tf.FixedLenFeature([28, 28], tf.float32),
    }
)

x_2 = example["image"]
# tfimage = (tfimage * 127) + 128.0 # to unnormalize
y_2 = [tf.cast(example['label'], tf.float32)]

W_conv1_2 = weight_variable_2([cv_all_size, cv_all_size, channels_jpg, cv_all_channels])
b_conv1_2 = bias_variable_2([cv_all_channels])
W_conv2_2 = weight_variable_2([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels * 2])
b_conv2_2 = bias_variable_2([cv_all_channels * 2])
# W_conv3_2 = weight_variable_2([cv_all_size, cv_all_size, cv_all_channels * 2, cv_all_channels * 4])
# b_conv3_2 = bias_variable_2([cv_all_channels * 4])
# W_conv4_2 = weight_variable_2([cv_all_size, cv_all_size, cv_all_channels * 4, cv_all_channels * 8])
# b_conv4_2 = bias_variable_2([cv_all_channels * 8])
# W_conv5_2 = weight_variable_2([cv_all_size, cv_all_size, cv_all_channels * 8, cv_all_channels * 16])
# b_conv5_2 = bias_variable_2([cv_all_channels * 16])

W_fc1_2 = weight_variable_2([last_img_size * last_img_size * cv_all_channels * 2, hidden])
b_fc1_2 = bias_variable_2([hidden])
W_fc2_2 = weight_variable_2([hidden, categories])
b_fc2_2 = bias_variable_2([categories])

x_r = tf.reshape(tf.cast(x_2, tf.float32), [-1,img_width,img_height,channels_jpg])

# conv
h_conv1_2 = tf.nn.relu(conv2d(x_r, W_conv1_2) + b_conv1_2)
h_lrn1_2 = conv2d_lrn(h_conv1_2)
h_pool1_2 = max_pool_2x2(h_lrn1_2)
h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
h_lrn2_2 = conv2d_lrn(h_conv2_2)
h_pool2_2 = max_pool_2x2(h_lrn2_2)
# h_conv3_2 = tf.nn.relu(conv2d(h_pool2_2, W_conv3_2) + b_conv3_2)
# h_lrn3_2 = conv2d_lrn(h_conv3_2)
# h_pool3_2 = max_pool_2x2(h_lrn3_2)
# h_conv4_2 = tf.nn.relu(conv2d(h_pool3_2, W_conv4_2) + b_conv4_2)
# h_lrn4_2 = conv2d_lrn(h_conv4_2)
# h_pool4_2 = max_pool_2x2(h_lrn4_2)
# h_conv5_2 = tf.nn.relu(conv2d(h_pool4_2, W_conv5_2) + b_conv5_2)
# h_lrn5_2 = conv2d_lrn(h_conv5_2)
# h_pool5_2 = max_pool_2x2(h_lrn5_2)

h_pool3_2 = aver_pool_7x7(h_pool2_2, 1)
# h_pool3_2 = max_pool_7x7(aver_pool_7x7(h_pool2_2, 1), 1)
h_pool_last_flat_2 = tf.reshape(h_pool3_2, [-1, last_img_size * last_img_size  * cv_all_channels * 2])

h_fc1_2 = tf.nn.relu(tf.matmul(h_pool_last_flat_2, W_fc1_2) + b_fc1_2)
pred_2 = tf.nn.softmax(tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2)
pred2_2 = tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2

cost_2 = tf.reduce_mean(-tf.reduce_sum(y_2 * tf.log(pred_2 + 1e-20), reduction_indices=[1]))
cost_1_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred2_2,labels=y_2))

correct_prediction_2 = tf.equal(tf.argmax(pred_2, 1), tf.argmax(y_2, 1))
accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))
pred_arg2_2 = tf.argmax(pred_2, 1)
y_arg2_2 = tf.argmax(y_2, 1)
auc2_2, update_op_auc2_2 = tf.contrib.metrics.streaming_auc(predictions=pred_2, labels=y_2, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)

init = tf.global_variables_initializer()
acc_total = 0.0

zero = 0
one = 0
two = 0
three = 0
four = 0
five = 0
six = 0
seven = 0
eight = 0
nine = 0

zero_2 = 1e-20
one_2 = 1e-20
two_2 = 1e-20
three_2 = 1e-20
four_2 = 1e-20
five_2 = 1e-20
six_2 = 1e-20
seven_2 = 1e-20
eight_2 = 1e-20
nine_2 = 1e-20

total_cost2 = 0
total_cost3 = 0
with tf.Session() as sess:
	sess.run(tf.local_variables_initializer())
	sess.run(init)
	sess.run([
	    W_conv1_2.assign(features["W_conv1"]),
	    b_conv1_2.assign(features["b_conv1"][0]),
	    W_conv2_2.assign(features["W_conv2"]),
	    b_conv2_2.assign(features["b_conv2"][0]),
	    # W_conv3_2.assign(features["W_conv3"]),
	    # b_conv3_2.assign(features["b_conv3"][0]),
	    # W_conv4_2.assign(features["W_conv4"]),
	    # b_conv4_2.assign(features["b_conv4"][0]),
	    # W_conv5_2.assign(features["W_conv5"]),
	    # b_conv5_2.assign(features["b_conv5"][0]),
	    W_fc1_2.assign(features["W_fc1"]),
	    b_fc1_2.assign(features["b_fc1"][0]),
	    W_fc2_2.assign(features["W_fc2"]),
	    b_fc2_2.assign(features["b_fc2"][0])
    ])
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for step in xrange(samples):
		print "step:",step
		prob,correct_prediction2,acc,pred_arg,y_arg,y_tes,update_op_auc,auc, cost2,cost3 = sess.run([pred_2,correct_prediction_2,accuracy_2,pred_arg2_2,y_arg2_2,y_2,update_op_auc2_2,auc2_2,cost_2,cost_1_2])
		prob = prob[0]
		y_arg = y_arg[0]
		pred_arg = pred_arg[0]
		correct_prediction2 = correct_prediction2[0]
		total_cost2 += cost2
		total_cost3 += cost3
		print "auc:", auc,"update_op_auc:",update_op_auc
		if correct_prediction2 == True:
			if y_arg == 0:
			    zero += 1
			elif y_arg == 1:
			    one += 1
			elif y_arg == 2:
			    two += 1
			elif y_arg == 3:
			    three += 1
			elif y_arg == 4:
			    four += 1
			elif y_arg == 5:
			    five += 1
			elif y_arg == 6:
			    six += 1
			elif y_arg == 7:
			    seven += 1
			elif y_arg == 8:
			    eight += 1
			elif y_arg == 9:
			    nine += 1
		if y_arg == y_arg:
			if y_arg == 0:
			    zero_2 += 1
			elif y_arg == 1:
			    one_2 += 1
			elif y_arg == 2:
			    two_2 += 1
			elif y_arg == 3:
			    three_2 += 1
			elif y_arg == 4:
			    four_2 += 1
			elif y_arg == 5:
			    five_2 += 1
			elif y_arg == 6:
			    six_2 += 1
			elif y_arg == 7:
			    seven_2 += 1
			elif y_arg == 8:
			    eight_2 += 1
			elif y_arg == 9:
			    nine_2 += 1
		acc_total += acc
	auc = auc2_2.eval()

	total_s = zero_2 + one_2 + two_2 + three_2 + four_2 + five_2 + six_2 + seven_2 + eight_2 + nine_2
	acc_label = (zero*100.0/zero_2 + one*100.0/one_2 + two*100.0/two_2 + three*100.0/three_2 +
	    four*100.0/four_2 + five*100.0/five_2 + six*100.0/six_2 + seven*100.0/seven_2 +
	    eight*100.0/eight_2 + nine*100.0/nine_2) / 10.0
	print
	print "auc:", auc
	print "accuracy:", round(acc_total*100.0/total_s,2),"total good int:",int(acc_total),"of",int(total_s)
	print "acc by label:", round(acc_label,2)
	print "cost2:", total_cost2*1.0/total_s, "cost3:", total_cost3*1.0/total_s
	print
	print "zero   ", round(zero*100.0/zero_2,2),"total good int:",zero, "of", int(zero_2)
	print "one    ", round(one*100.0/one_2,2),"total good int:",one, "of", int(one_2)
	print "two    ", round(two*100.0/two_2,2),"total good int:",two, "of", int(two_2)
	print "three  ", round(three*100.0/three_2,2),"total good int:",three, "of", int(three_2)
	print "four   ", round(four*100.0/four_2,2),"total good int:",four, "of", int(four_2)
	print "five   ", round(five*100.0/five_2,2),"total good int:",five, "of", int(five_2)
	print "six    ", round(six*100.0/six_2,2),"total good int:",six, "of", int(six_2)
	print "seven  ", round(seven*100.0/seven_2,2),"total good int:",seven, "of", int(seven_2)
	print "eight  ", round(eight*100.0/eight_2,2),"total good int:",eight, "of", int(eight_2)
	print "nine   ", round(nine*100.0/nine_2,2),"total good int:",nine, "of", int(nine_2)
	print

	coord.request_stop()
	coord.join(threads)
	sess.close()