import tensorflow as tf

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def weight_variable_2(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=False)

def bias_variable_2(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def lrn(x):
	return tf.nn.lrn(x)

def aver_pool(x, n):
	return tf.nn.avg_pool(x, ksize=[1, n, n, 1], strides=[1, 1, 1, 1], padding='SAME')

def aver_pool_7x7(x, s):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, s, s, 1], padding='SAME')