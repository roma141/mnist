import tensorflow as tf
import numpy as np
from apiDeepLearning import *
import scipy.io
from params import param
import time
import os

print "initialize"
directory = "/media/roma141/Alma 2TB/torrents/mnist/normal-x8"
directory_test = "/media/roma141/Alma 2TB/torrents/mnist/normal"

img_path = directory + "/train-tfrecord/"
files = os.listdir(img_path)
img_queue = []
full_cost = {}
full_acc = {}
for file in files:
    img_queue.append(img_path + file)
    full_cost[file.split(".")[0]] = 1e1
    full_acc[file.split(".")[0]] = 0.0

img_path_test = directory_test + "/test-tfrecord/"
files = os.listdir(img_path_test)
samples = len(files)
img_queue_test = []
for file in files:
    img_queue_test.append(img_path_test + file)

training_epochs = 8000000 + 1
display_step = 1

parameters = param()
hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
learning_rate = parameters["learning_rate"]
# dropout = parameters["dropout"]
dropout = 1.0
# beta = 0.001
beta = 0.17 # only y
# beta = 0.01 # softmax on y
save_epoch = 1000
cv_all_size = 5
cv_all_channels = 8
last_img_size = 7
batch_size = 100
channels_jpg = 1
mat_name_file = "_conv2_chan_" + str(cv_all_channels)
head_mat_name = "resp_tfrecord"
to_break = 10000

### testable net
filename_queue_img_test = tf.train.string_input_producer(img_queue_test, shuffle=False)

image_reader_test = tf.TFRecordReader()
_, data_test = image_reader_test.read(filename_queue_img_test)

example_test = tf.parse_single_example(
    data_test,
    features = {
        'label': tf.FixedLenFeature([10], tf.float32),
        'image': tf.FixedLenFeature([28, 28], tf.float32),
    }
)

x_2 = example_test["image"]
# tfimage = (tfimage * 127) + 128.0 # to unnormalize
y_2 = [tf.cast(example_test['label'], tf.float32)]

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


### trainable net
filename_queue = tf.train.string_input_producer(img_queue, shuffle=True)

reader = tf.TFRecordReader()
tfname, data = reader.read(filename_queue)
    
example = tf.parse_single_example(
    data,
    features = {
        'label': tf.FixedLenFeature([10], tf.float32),
        'image': tf.FixedLenFeature([28, 28], tf.float32),
    }
)

tfimage = example["image"]
# tfimage = (tfimage * 127) + 128.0 # to unnormalize
tflabel = tf.cast(example['label'], tf.float32)

# min_after_dequeue + 3 * batch_size
x, y, n = tf.train.shuffle_batch(
    [tfimage, tflabel, tfname], batch_size = batch_size, 
    capacity = 80000,
    min_after_dequeue = 20000)

# x = tf.cast(x, tf.float32)
# y = tf.cast(y, tf.float32)
keep_prob = tf.placeholder(tf.float32)

# x = tf.image.random_brightness(xxx, max_delta=32. / 255.)
# # x = tf.image.random_saturation(x, lower=0.5, upper=1.5)
# # x = tf.image.random_hue(x, max_delta=0.2)
# x = tf.image.random_contrast(xxx, lower=0.5, upper=1.5)
# x = tf.image.random_flip_left_right(xxx)

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
# W_conv8 = weight_variable([cv_all_size, cv_all_size, cv_all_channels * 64, cv_all_channels * 128])
# b_conv8 = bias_variable([cv_all_channels * 128])
# W_conv9 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv9 = bias_variable([cv_all_channels])
# W_conv10 = weight_variable([cv_all_size, cv_all_size, cv_all_channels, cv_all_channels])
# b_conv10 = bias_variable([cv_all_channels])

W_fc1 = weight_variable([last_img_size * last_img_size * (cv_all_channels * 2), hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])

x_r = tf.reshape(x, [-1, img_width, img_height,channels_jpg])
# x = tf.reshape(x, [-1])

# conv
h_conv1 = tf.nn.relu(conv2d(x_r, W_conv1) + b_conv1)
h_lrn1 = conv2d_lrn(h_conv1)
h_pool1 = max_pool_2x2(h_lrn1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_lrn2 = conv2d_lrn(h_conv2)
h_pool2 = max_pool_2x2(h_lrn2)
# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_lrn3 = conv2d_lrn(h_conv3)
# h_pool3 = max_pool_2x2(h_lrn3)
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_lrn4 = conv2d_lrn(h_conv4)
# h_pool4 = max_pool_2x2(h_lrn4)
# h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
# h_lrn5 = conv2d_lrn(h_conv5)
# h_pool5 = max_pool_2x2(h_lrn5)
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
h_pool_last_flat = tf.reshape(h_pool3, [-1, last_img_size * last_img_size  * (cv_all_channels * 2)])

# full conected
h_fc1 = tf.nn.relu(tf.matmul(h_pool_last_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print "x", x
print "y", y
print "pred", pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2
# print "h_pool3", h_pool3

# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)) +
#     (beta * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_conv1) + 
#         tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_conv4) +
#         tf.nn.l2_loss(W_conv5))))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y) +
#     beta * tf.nn.l2_loss(W_fc1) + beta * tf.nn.l2_loss(W_fc2) + beta * tf.nn.l2_loss(W_conv1) + beta * tf.nn.l2_loss(W_conv2)
#      + beta * tf.nn.l2_loss(W_conv3) + beta * tf.nn.l2_loss(W_conv4) + beta * tf.nn.l2_loss(W_conv5))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=tf.nn.softmax(y)))
# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=tf.nn.softmax(y))) +
#     beta * tf.nn.l2_loss(W_fc1) + beta * tf.nn.l2_loss(W_fc2) + beta * tf.nn.l2_loss(W_conv1) + beta * tf.nn.l2_loss(W_conv2)
#      + beta * tf.nn.l2_loss(W_conv3) + beta * tf.nn.l2_loss(W_conv4) + beta * tf.nn.l2_loss(W_conv5))
# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)) + tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y,pred)),1)))/2.0
# _, roc = tf.contrib.metrics.streaming_auc(logits=pred,labels=y, weights=None, num_thresholds=200, metrics_collections=None, updates_collections=None, curve='ROC', name=None)
# cost = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y)) + (1 - roc))/2.0
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))
# cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=pred,labels=y, 4))
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=tf.argmax(y, 1)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_arg2 = tf.argmax(y, 1)

init = tf.global_variables_initializer()

features = {}
save = 100000.0
save2 = 0.0
breaking_count = 0
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

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

    # sleep = 0
    # print "queue sleep in seg:", sleep
    # time.sleep(sleep)
    # print "end queue sleep"

    # Training cycle
    print "learning..."
    for epoch in xrange(training_epochs):
        _, c, acc, fname = sess.run([optimizer, cost, accuracy,n], feed_dict={keep_prob: dropout})
        name = fname[0].split("/")[-1].split(".")[0]
        full_cost[name] = c
        full_acc[name] = acc

        # print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"dropout:",dropout,"bad acc:", round(acc*100.0,2),"%"
            
        if epoch%save_epoch == 0:
            cf = sum([full_cost[rol] for rol in full_cost]) / len(full_cost) * 1.0
            t_acc = sum([full_acc[rol] for rol in full_acc]) / len(full_acc) * 1.0
            print "Ep:", '%04d' % (epoch),"f-c:","{:.9f}".format(cf), "c:", "{:.9f}".format(c),"f-b-acc:", "{:02.2f}".format(t_acc*100.0),"%","dropout:",dropout,"b-acc:", round(acc*100.0,2),"%"
            
            sess.run([
                W_conv1_2.assign(W_conv1.eval()),
                b_conv1_2.assign(b_conv1.eval()),
                W_conv2_2.assign(W_conv2.eval()),
                b_conv2_2.assign(b_conv2.eval()),
                # W_conv3_2.assign(W_conv3.eval()),
                # b_conv3_2.assign(b_conv3.eval()),
                # W_conv4_2.assign(W_conv4.eval()),
                # b_conv4_2.assign(b_conv4.eval()),
                # W_conv5_2.assign(W_conv5.eval()),
                # b_conv5_2.assign(b_conv5.eval()),
                W_fc1_2.assign(W_fc1.eval()),
                b_fc1_2.assign(b_fc1.eval()),
                W_fc2_2.assign(W_fc2.eval()),
                b_fc2_2.assign(b_fc2.eval())
            ])

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

            for step in xrange(samples):
                # print "step:",step
                prob,correct_prediction2,acc,pred_arg,y_arg,y_tes,update_op_auc,auc, cost2,cost3 = sess.run([pred_2,correct_prediction_2,accuracy_2,pred_arg2_2,y_arg2_2,y_2,update_op_auc2_2,auc2_2,cost_2,cost_1_2])
                prob = prob[0]
                y_arg = y_arg[0]
                pred_arg = pred_arg[0]
                correct_prediction2 = correct_prediction2[0]
                total_cost2 += cost2
                total_cost3 += cost3
                # print "auc:", auc,"update_op_auc:",update_op_auc
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
            # print "auc:", auc
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

            print "best cost on test", save,"-- best on acc", save2
            if save2 < (acc_total*100.0/total_s):
                save2 = acc_total*100.0/total_s
                print "saved acc"
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
                scipy.io.savemat(str(head_mat_name) + str(mat_name_file) + "_acc", features, do_compression=True)
                if breaking_count > to_break/0.5:
                    to_break += to_break
                breaking_count = 0
            if save > (total_cost3*1.0/total_s):
                save = (total_cost3*1.0/total_s)
                print "saved cost"
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
                scipy.io.savemat(str(head_mat_name) + str(mat_name_file), features, do_compression=True)
                if breaking_count > to_break/0.5:
                    to_break += to_break
                breaking_count = 0
            else:
                if breaking_count > to_break:
                    print "breaking"
                    break
                else:
                    print "waiting if get better...try:",breaking_count
                    breaking_count += 1
                    continue
                print "breaking"
                break
                print "using back"
                sess.run([
                    W_conv1.assign(features["W_conv1"]),
                    b_conv1.assign(features["b_conv1"]),
                    W_conv2.assign(features["W_conv2"]),
                    b_conv2.assign(features["b_conv2"]),
                    W_conv3.assign(features["W_conv3"]),
                    b_conv3.assign(features["b_conv3"]),
                    W_conv4.assign(features["W_conv4"]),
                    b_conv4.assign(features["b_conv4"]),
                    W_conv5.assign(features["W_conv5"]),
                    b_conv5.assign(features["b_conv5"]),
                    W_fc1.assign(features["W_fc1"]),
                    b_fc1.assign(features["b_fc1"]),
                    W_fc2.assign(features["W_fc2"]),
                    b_fc2.assign(features["b_fc2"])
                ])
        
    print "Optimization Finished!"

    coord.request_stop()
    coord.join(threads)
    sess.close()


# print "epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout"
# print ("    %s      %s        %s       %s          %s            %s        %s        %s      %s      %s" 
#     % (training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"]
#        , parameters["cv2_channels"], parameters["hidden"], parameters["img_height"], parameters["img_width"], parameters["dropout"]))

print str(head_mat_name) + str(mat_name_file)
print "end"