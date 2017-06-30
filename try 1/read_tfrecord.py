import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

directory = "/media/roma141/Alma 2TB/torrents/mnist/normal-x8"

path = directory + "/train-tfrecord/"
path = directory + "/test-tfrecord/"
files = [path + file for file in os.listdir(path)]
samples = len(files)
# print files
print samples

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

tfimage = example["image"]
# tfimage = (tfimage * 127) + 128.0 # to unnormalize
tflabel = tf.cast(example['label'], tf.float32)


init = tf.global_variables_initializer()
with tf.Session() as sess:
	# sess.run(tf.local_variables_initializer())
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for step in xrange(samples):
		image, label, name = sess.run([tfimage, tflabel, tfname])

		n = name.split("/")[-1:][0].split(".")[0]
		print n, label
		print image.shape
		plt.gray()
		plt.imshow(image)
		plt.show()
		# if 3 > 1:
		# 	break

	coord.request_stop()
	try: 
	    coord.join(threads)
	except:
	    pass
	sess.close()

print "end"