import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import learn, layers
from tf_deep import TFDeep

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')
mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)

model = TFDeep([784, 10], ldir='mnist_784_10_l2_10', l2=10)
model.fit(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels, 128, 10000)
