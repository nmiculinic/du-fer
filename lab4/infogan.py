import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d_transpose, conv_2d
from tflearn.layers.normalization import batch_normalization
# from tflearn.activations import leaky_relu
import seaborn as sns

sns.set_style("dark")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


class DCGAN():
    def __init__(self, batch_size, z):
        self.z = tf.random_normal(shape=[batch_size, z])
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])

        with tf.variable_scope("generator"):
            self.g_log, self.g = self.generator()

        with tf.variable_scope("disriminator"):
            self.d_log_real, self.d_real = self.disriminator(self.X)

        with tf.variable_scope("disriminator", reuse=True):
            self.d_log_fake, self.d_fake = self.disriminator(self.g)

        self.loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_log_fake,
                tf.ones_like(self.d_log_fake)
            )
        )

        self.loss_d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_log_fake,
                tf.zeros_like(self.d_log_fake)
            )
        )

        self.loss_d_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_log_real,
                tf.ones_like(self.d_log_real)
            )
        )

        self.loss_d = self.loss_d_fake + self.loss_d_real

        self.gen_summ = tf.summary.merge([
            tf.summary.scalar("loss_g", self.loss_g)
        ])

        self.dis_summ = tf.summary.merge([
            tf.summary.scalar("loss_d_fake", self.loss_d_fake),
            tf.summary.scalar("loss_d_real", self.loss_d_real),
            tf.summary.scalar("loss_d", self.loss_d),
        ])

        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='disriminator'
        )
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator'
        )

        self.train_d = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
            self.loss_d,
            var_list=d_vars
        )
        #
        self.train_g = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
            self.loss_g,
            var_list=g_vars
        )

    def generator(self):
        net = self.z
        net = fully_connected(net, 49)
        net = tf.reshape(net, [tf.shape(net)[0], 7, 7, 1])

        # net = conv_2d_transpose(net, 16, 3, [16, 16], scope="l1")
        # net = batch_normalization(net, scope="bn_1")
        # net = tf.nn.relu(net)

        # net = conv_2d_transpose(net, 8, 3, [24, 24], scope="l2")
        # net = batch_normalization(net, scope="bn_2")
        # net = tf.nn.relu(net)

        net = conv_2d_transpose(net, 1, 3, [28, 28], scope="l3")
        return net, tf.nn.sigmoid(net)

    def disriminator(self, net):
        net = conv_2d(net, 4, 3, strides=2, scope='l1')
        net = leaky_relu(net)

        net = conv_2d(net, 4, 3, strides=2, scope='l2', bias=False)  # BN handles bias
        net = batch_normalization(net, scope='bn_l2')
        net = leaky_relu(net)

        net = conv_2d(net, 4, 3, strides=2, scope='l3', bias=False)
        net = batch_normalization(net, scope='bn_l3')
        net = leaky_relu(net)

        net = fully_connected(net, 1)
        return net, tf.nn.sigmoid(net)


if __name__ == "__main__":
    model = DCGAN(32, 50)
