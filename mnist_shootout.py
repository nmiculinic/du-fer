#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf_deep import TFDeep
import numpy as np

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

for l2 in np.logspace(-2, 2, 5):
    print("Training l2 = %.3f" % l2)
    model = TFDeep([784, 10], ldir='mnist_784_10_l2_%.3f' % l2, l2=l2)
    model.fit(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, 128, 500)
