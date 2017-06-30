import tensorflow as tf
import numpy as np
import scipy.io

def weight_variable(shape):
    # return tf.Variable(tf.random_normal(shape, stddev=0.35))
    # return tf.Variable(tf.random_normal(shape, stddev=0.1, mean=0.0))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, mean=0.0))
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.35))

def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))
    # return tf.Variable(tf.constant(0.1, shape=shape))

def weight_variable_2(shape):
    # return tf.Variable(tf.random_normal(shape, stddev=0.35))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, mean=0.0), trainable=False)
    # return tf.Variable(tf.truncated_normal(shape, stddev=0.35))

def bias_variable_2(shape):
    # return tf.Variable(tf.zeros(shape))
    return tf.Variable(tf.constant(0.1, shape=shape), trainable=False)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')

# def conv2d(x, W, a):
#     if a < 5:
#         return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#     elif a < 8:
#         return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
#     elif a > 8:
#         return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')

def conv2d_2(x, W, a):
    if a < 6:
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    elif a < 8:
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
    elif a > 8:
        return tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x, s):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, s, s, 1], padding='SAME')

def max_pool_7x7(x, s):
    return tf.nn.max_pool(x, ksize=[1, 7, 7, 1], strides=[1, s, s, 1], padding='SAME')

def aver_pool_5x5(x, s):
    return tf.nn.avg_pool(x, ksize=[1, 5, 5, 1], strides=[1, s, s, 1], padding='SAME')

def aver_pool_7x7(x, s):
    return tf.nn.avg_pool(x, ksize=[1, 7, 7, 1], strides=[1, s, s, 1], padding='SAME')

def conv2d_lrn(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def lrn(x):
    # return tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    return tf.nn.lrn(x)