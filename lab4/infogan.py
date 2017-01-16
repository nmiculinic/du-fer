import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.layers.core import fully_connected
from tflearn.layers.conv import conv_2d_transpose, conv_2d
import seaborn as sns
import numpy as np
import shutil
import os
import tflearn
import socket
import logging

log_fmt = '[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
sns.set_style("dark")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
repo_root = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))


def leaky_relu(x):
    return tf.maximum(x, 0.2 * x)


def default_generator(net):
    net = fully_connected(net, 7 * 7 * 16, scope='fc')
    net = tf.reshape(net, [tf.shape(net)[0], 7, 7, 16])
    net = conv_2d_transpose(net, 1, 5, [28, 28], scope="l0", strides=4, bias=True)
    return net


def default_dq_common(net):
    net = conv_2d(net, 64, 5, strides=2, scope='l0')
    net = leaky_relu(net)
    return net


class InfoGAN():
    def __init__(self, name, generator_fn=None, dq_common_fn=None, batch_size=128, num_z=100, n_bernulli=10, n_gauss=3, l_bernulli=1.0, l_gauss=0.5, clear_logdir=False):
        self.log_dir = os.path.join(
            repo_root, 'log', socket.gethostname(), name)
        self.logger = logging.getLogger(name)
        if os.path.exists(self.log_dir):
            self.logger.warn("Logdir exists!")
        if clear_logdir:
            self.logger.info("Clearing logdir")
            shutil.rmtree(self.log_dir, ignore_errors=True)

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.batch_size = tf.placeholder_with_default(batch_size, [])
            self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])  # MNIST

            with tf.name_scope("gen"):
                self.n_bernulli = n_bernulli
                self.n_gauss = n_gauss
                self.n_c = n_gauss + n_bernulli

                self.c_bernulli = tf.to_float(
                    tf.random_uniform([self.batch_size, n_bernulli]) < 0.5)
                self.c_gauss = tf.random_normal([self.batch_size, n_gauss])

                self.z = tf.random_uniform(shape=[self.batch_size, num_z])
                self.z = tf.concat_v2([self.c_gauss, self.c_bernulli, self.z], 1)

            with tf.variable_scope("generator"):
                if generator_fn is None:
                    self.logger.warn("Using default generator function")
                    generator_fn = default_generator
                self.generator = generator_fn

                self.g_log = self.generator(self.z)
                self.g = tf.nn.sigmoid(self.g_log)

            with tf.variable_scope("disriminator"):
                if dq_common_fn is None:
                    self.logger.warn("Using default dq_common function")
                    dq_common_fn = default_dq_common
                self.dq_common = dq_common_fn
                self.d_log_real = fully_connected(self.dq_common(self.X), 1, scope='fc')
                self.d_real = tf.nn.sigmoid(self.d_log_real)

            with tf.variable_scope("disriminator", reuse=True):
                qnet = self.dq_common(self.g)
                self.d_log_fake = fully_connected(qnet, 1, scope='fc')
                self.d_fake = tf.nn.sigmoid(self.d_log_fake)

            with tf.variable_scope("q"):
                if n_bernulli > 0:
                    self.loss_q_ber = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            fully_connected(qnet, n_bernulli, scope='bernulli'),
                            self.c_bernulli
                        )
                    )  # Average loss per c_bernulli
                else:
                    self.logger.warn("No Bernulli variables")
                    self.loss_q_ber = tf.constant(0.0)

                if n_gauss > 0:
                    qc_mean = fully_connected(qnet, n_gauss, scope='normal_mu')
                    qc_sigma = tf.abs(fully_connected(
                        qnet, n_gauss, scope='normal_sigma')
                    )

                    dist = tf.contrib.distributions.Normal(qc_mean, qc_sigma)
                    qc_log_normal = dist.log_pdf(self.c_gauss)
                    # -log jer minimiziram
                    self.loss_q_gauss = tf.reduce_mean(-qc_log_normal * self.c_gauss)
                else:
                    self.logger.warn("No Gauss variables")
                    self.loss_q_gauss = tf.constant(0.0)

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

            with tf.name_scope("train"):
                self.d_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='disriminator'
                )
                self.g_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='generator'
                )
                self.q_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,
                    scope='q'
                )

                self.train_d = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
                    self.loss_d,
                    var_list=self.d_vars
                )
                self.train_gq = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(
                    self.loss_g + self.loss_q,
                    var_list=[*self.g_vars, *self.q_vars]
                )

                self.global_step = tf.get_variable("global_step", initializer=tf.constant_initializer(), shape=(), dtype=tf.int32)
                self.inc_global_step = tf.assign_add(self.global_step, 1)

            self.init_op = tf.global_variables_initializer()
            self.logger.info("Graph construction finished")
            self.logger.info("log_dir %s", self.log_dir)

            self.train_writer = tf.summary.FileWriter(
                self.log_dir, graph=self.graph, flush_secs=60)
            self.saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=1,
            )

    def train_loop(self, summary_every=20, save_every=1000):
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
            self.sess.run(self.train_gq)
            self.sess.run(self.train_gq)
            self.sess.run(self.train_d, feed_dict=fd)
            self.sess.run(self.train_gq)
            self.sess.run(self.train_gq)

        if step % save_every == 0:
            self.save()

        self.sess.run(self.inc_global_step)

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
            self.logger.warn("%4d No Checkpoints found", iter_step)
        else:
            self.saver.restore(self.sess, checkpoint)
            iter_step = self.get_global_step()
            self.logger.info("%4d Restored to checkpoint %s",
                             iter_step, checkpoint)
        return iter_step

    def init_session(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)
        # self.graph.finalize()
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

        with model.graph.as_default():
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                model.logger.debug(var.name)

            model.logger.debug("G phase vars")
            for var in [*model.g_vars, *model.q_vars]:
                model.logger.debug(var.name)

            model.logger.debug("D phase vars")
            for var in model.d_vars:
                model.logger.debug(var.name)

        for i in range(model.get_global_step() + 1, 101):
            model.train_loop(summary_every=10)
            model.graph.finalize()
    finally:
        model.close_session()
