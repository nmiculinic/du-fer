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


class InfoGAN():
    def __init__(self, batch_size, num_z=20, n_bernulli=10, n_gauss=3, l_bernulli=1.0, l_gauss=0.5):
        with tf.name_scope("gen"):
            self.n_bernulli = n_bernulli
            self.n_gauss = n_gauss
            self.n_c = n_gauss + n_bernulli

            self.c_bernulli = tf.to_float(tf.random_uniform([batch_size, n_bernulli]) < 0.5)

            self.c_gauss = tf.random_normal([batch_size, n_gauss])

            self.c = tf.concat_v2([self.c_gauss, self.c_bernulli], 1)

            self.z = tf.random_normal(shape=[batch_size, num_z])
            self.z = tf.concat_v2([self.c, self.z], 1)

        print(self.z.get_shape())

        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])

        with tf.variable_scope("generator"):
            self.g_log, self.g = self.generator()

        with tf.variable_scope("disriminator"):
            self.d_log_real, self.d_real = self.disriminator(self.X)

        with tf.variable_scope("disriminator", reuse=True):
            self.d_log_fake, self.d_fake = self.disriminator(self.g)
            qnet = self.dq_common(self.g)

        with tf.variable_scope("q"):
            self.loss_q_ber = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    fully_connected(qnet, n_bernulli, scope='bernulli'),
                    self.c_bernulli
                )
            )  # Average loss per c_bernulli

            qc_mean = fully_connected(qnet, n_gauss, scope='normal_mu')
            qc_sigma = tf.abs(fully_connected(qnet, n_gauss, scope='normal_sigma'))

            dist = tf.contrib.distributions.Normal(qc_mean, qc_sigma)
            qc_log_normal = dist.log_pdf(self.c_gauss)
            print("qc_normal", qc_log_normal.get_shape())
            self.loss_q_gauss = tf.reduce_mean(-qc_log_normal * self.c_gauss)  # -log jer minimiziram

        self.loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.d_log_fake,
                tf.ones_like(self.d_log_fake)
            )
        )

        self.loss_q =\
            l_gauss * self.loss_q_gauss +\
            l_bernulli * self.loss_q_ber

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

        with tf.name_scope("loss"):
            self.gen_summ = tf.summary.merge([
                tf.summary.scalar("g", self.loss_g),
                tf.summary.scalar("q", self.loss_q),
                tf.summary.scalar("q_gauss", self.loss_q_gauss),
                tf.summary.scalar("q_ber", self.loss_q_ber)
            ])

            self.dis_summ = tf.summary.merge([
                tf.summary.scalar("d_fake", self.loss_d_fake),
                tf.summary.scalar("d_real", self.loss_d_real),
                tf.summary.scalar("d", self.loss_d),
            ])

        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='disriminator'
        )
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='generator'
        )

        q_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='q'
        )

        self.train_d = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
            self.loss_d,
            var_list=d_vars
        )

        self.train_gq = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
            self.loss_g + self.loss_q,
            var_list=[*g_vars, *q_vars]
        )

    def generator(self):
        net = self.z
        net = fully_connected(net, 49, scope='fc')
        net = tf.reshape(net, [tf.shape(net)[0], 7, 7, 1])

        # net = conv_2d_transpose(net, 16, 3, [16, 16], scope="l1")
        # net = batch_normalization(net, scope="bn_1")
        # net = tf.nn.relu(net)

        # net = conv_2d_transpose(net, 8, 3, [24, 24], scope="l2")
        # net = batch_normalization(net, scope="bn_2")
        # net = tf.nn.relu(net)

        net = conv_2d_transpose(net, 1, 3, [28, 28], scope="l3")
        return net, tf.nn.sigmoid(net)

    def dq_common(self, net):
        net = conv_2d(net, 4, 3, strides=2, scope='l1')
        net = leaky_relu(net)

        net = conv_2d(net, 4, 3, strides=2, scope='l2', bias=False)  # BN handles bias
        net = batch_normalization(net, scope='bn_l2')
        net = leaky_relu(net)

        net = conv_2d(net, 4, 3, strides=2, scope='l3', bias=False)
        net = batch_normalization(net, scope='bn_l3')
        net = leaky_relu(net)
        return net



    def disriminator(self, net):
        net = self.dq_common(net)
        net = fully_connected(net, 1, scope='fc')
        return net, tf.nn.sigmoid(net)


if __name__ == "__main__":
    model = InfoGAN(32)
