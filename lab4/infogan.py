import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d_transpose, conv_2d
from tflearn.layers.normalization import batch_normalization
import seaborn as sns
import numpy as np
import shutil
import os
import tflearn
import socket
import logging

log_fmt = '[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
sns.set_style("dark")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
repo_root = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


class InfoGAN():

    def __init__(self, name, batch_size=32, num_z=100, n_bernulli=10, n_gauss=3, l_bernulli=1.0, l_gauss=0.5, clear_logdir=False):
        self.log_dir = os.path.join(
            repo_root, 'log', socket.gethostname(), name)
        self.logger = logging.getLogger(name)

        self.graph = tf.Graph()
        with self.graph.as_default():
            batch_size = tf.placeholder_with_default(batch_size, [])
            self.batch_size = batch_size

            with tf.name_scope("gen"):
                self.n_bernulli = n_bernulli
                self.n_gauss = n_gauss
                self.n_c = n_gauss + n_bernulli

                self.c_bernulli = tf.to_float(
                    tf.random_uniform([batch_size, n_bernulli]) < 0.5)
                self.c_gauss = tf.random_normal([batch_size, n_gauss])

                self.z = tf.random_uniform(shape=[batch_size, num_z])
                self.z = tf.concat_v2([self.c_gauss, self.c_bernulli, self.z], 1)

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
                qc_sigma = tf.abs(fully_connected(
                    qnet, n_gauss, scope='normal_sigma')
                )

                dist = tf.contrib.distributions.Normal(qc_mean, qc_sigma)
                qc_log_normal = dist.log_pdf(self.c_gauss)
                # -log jer minimiziram
                print("qc_normal", qc_log_normal.get_shape())
                self.loss_q_gauss = tf.reduce_mean(-qc_log_normal * self.c_gauss)

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
                    tf.summary.scalar("q_ber", self.loss_q_ber),
                    tf.summary.image("gen_example", self.g, max_outputs=10)
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

            self.train_d = tf.train.AdamOptimizer().minimize(
                self.loss_d,
                var_list=d_vars
            )
            self.train_gq = tf.train.AdamOptimizer().minimize(
                self.loss_g + self.loss_q,
                var_list=[*g_vars, *q_vars]
            )

            self.global_step = tf.get_variable("global_step", initializer=tf.constant_initializer(), shape=(), dtype=tf.int32)
            self.inc_global_step = tf.assign_add(self.global_step, 1)
            self.init_op = tf.global_variables_initializer()

            self.logger.info("Graph construction finished")
            self.logger.info("log_dir %s", self.log_dir)
            if os.path.exists(self.log_dir):
                self.logger.warn("Logdir exists!")
            if clear_logdir:
                self.logger.info("Clearing logdir")
                shutil.rmtree(self.log_dir, ignore_errors=True)
            self.train_writer = tf.summary.FileWriter(
                self.log_dir, graph=tf.get_default_graph(), flush_secs=60)
            self.saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=1,
            )

    def train_loop(self, summary_every=100, save_every=1000):
        step = self.get_global_step()
        with self.graph.as_default():
            tflearn.is_training(True, session=self.sess)
        X, _ = mnist.train.next_batch(self.sess.run(self.batch_size))
        fd = {
            self.X: X.reshape(-1, 28, 28, 1)
        }
        if step % summary_every == 0:
            _, summ, loss_dis = self.sess.run([self.train_d, self.dis_summ, self.loss_d], feed_dict=fd)
            self.train_writer.add_summary(summ, global_step=step)
            self.sess.run(self.train_gq)
            _, summ, loss_gen, loss_q = self.sess.run([self.train_gq, self.gen_summ, self.loss_g, self.loss_q])
            self.train_writer.add_summary(summ, global_step=step)
            self.logger.info("%4d dis %7.5f, gen %7.5f, q %7.5f", step, loss_dis, loss_gen, loss_q)
            self.train_writer.flush()
        else:
            self.sess.run(self.train_d, feed_dict=fd)
            self.sess.run(self.train_gq)
            self.sess.run(self.train_gq)

        if step % save_every == 0:
            self.save()

        self.sess.run(self.inc_global_step)

    def generator(self):
        net = self.z
        net = fully_connected(net, 20 * 49, scope='fc')
        net = tf.reshape(net, [tf.shape(net)[0], 7, 7, 20])

        net = conv_2d_transpose(
            net, 32, 3, [14, 14], strides=2, scope="l1", bias=False)
        net = batch_normalization(net, scope="bn_1")
        net = tf.nn.relu(net)

        net = conv_2d_transpose(net, 16, 3, [14, 14], scope="l2", bias=False)
        net = batch_normalization(net, scope="bn_2")
        net = tf.nn.relu(net)

        net = conv_2d_transpose(net, 1, 3, [28, 28], strides=2, scope="l3")
        return net, tf.nn.sigmoid(net)

    def dq_common(self, net):
        net = conv_2d(net, 4, 3, strides=2, scope='l1')
        net = leaky_relu(net)

        net = conv_2d(net, 4, 3, strides=2, scope='l2',
                      bias=False)  # BN handles bias
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

    def __img_from_data(self, pics, osize):
        from PIL import Image
        rows, cols, *_ = pics.shape
        sol = Image.new("RGB", (rows * osize, cols * osize))
        for i in range(rows):
            for j in range(cols):
                sol.paste(
                    Image.fromarray(
                        np.uint8(pics[i][j] * 255)).resize(
                            (osize, osize),
                            Image.NEAREST
                    ),
                    (osize * i, osize * j)
                )
        return sol

    def jupyter_sample_widgets(self, rows=4, cols=6, osize=28 * 4):
        from ipywidgets import interact

        with self.graph.as_default():
            tflearn.is_training(False, session=self.sess)

        def f(**kwargs):
            c_bernulli = np.array([
                kwargs["b%d" % i] for i in range(self.n_bernulli)
            ])
            c_gauss = np.array([
                kwargs["g%d" % i] for i in range(self.n_gauss)
            ])

            pics = self.sess.run(self.g, feed_dict={
                self.c_gauss: np.tile(c_gauss, (rows * cols, 1)),
                self.c_bernulli: np.tile(c_bernulli, (rows * cols, 1)),
                self.batch_size: rows * cols
            })
            pics = pics.reshape(rows, cols, 28, 28)
            return self.__img_from_data(pics, osize)

        return interact(
            f,
            **{("b%d" % i): True for i in range(self.n_bernulli)},
            **{("g%d" % i): (-3.0, 3.0) for i in range(self.n_gauss)},
            __manual=True
        )

    def get_global_step(self):
        return self.sess.run(self.global_step)

    def save(self):
        self.saver.save(
            self.sess,
            os.path.join(self.log_dir, 'model.ckpt'),
            global_step=self.get_global_step()
        )
        pic = self.__img_from_data(
            self.sess.run(self.g, feed_dict={self.batch_size: 12 * 8}).reshape(12, 8, 28, 28),
            28 * 4
        )
        fpath = os.path.join(self.log_dir, "examples%d.png" % self.get_global_step())
        pic.save(fpath)

        self.logger.info("%4d Saved generated images to %s", self.get_global_step(), fpath)
        self.logger.info("%4d saved checkpoing", self.get_global_step())

    def restore(self, checkpoint=None, must_exist=False):
        """
            Args:
                checkpoint: filename to restore, default to last checkpoint
        """
        checkpoint = checkpoint or tf.train.latest_checkpoint(self.log_dir)
        if checkpoint is None:
            if must_exist:
                raise ValueError("No checkpoints found")
            iter_step = self.get_global_step()
            self.logger.info("%4d Restored to checkpoint %s" %
                             (iter_step, checkpoint))
        else:
            self.saver.restore(self.sess, checkpoint)
            iter_step = self.get_global_step()
            self.logger.info("%4d Restored to checkpoint %s" %
                             (iter_step, checkpoint))
        return iter_step

    def init_session(self):
        self.sess = tf.Session(graph=self.graph)
        self.graph.finalize()
        self.sess.run(self.init_op)
        self.logger.info("Created session")

    def close_session(self):
        self.sess.close()
        self.train_writer.close()
        self.logger.info("Closed session")


if __name__ == "__main__":
    model = InfoGAN("model_1", clear_logdir=True)
    try:
        model.init_session()
        model.restore()
        for i in range(model.get_global_step() + 1, 101):
            model.train_loop(summary_every=10)
    finally:
        model.close_session()
