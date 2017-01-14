import tensorflow as tf
from PIL import Image
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")


def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


class RBM():

    def __init__(self, input_layer, num_out, inbias=None, outbias=None, gibbs_sampling_steps=1, alpha=0.1):
        n_in = input_layer.get_shape()[1]
        self.W = tf.get_variable(
            "W",
            shape=[n_in, num_out],
            initializer=tf.contrib.layers.xavier_initializer()
        )

        self.b_in = inbias or tf.get_variable(
            "b_in", initializer=tf.zeros_initializer(shape=[n_in]))

        self.b_out = outbias or tf.get_variable(
            "b_out", initializer=tf.zeros_initializer(shape=[num_out]))

        v0_prob = input_layer
        # batch_size x num_hidden
        h0_prob = tf.nn.sigmoid(tf.matmul(v0_prob, self.W) + self.b_out)
        h0 = sample_prob(h0_prob)

        self.out_prob = h0_prob
        self.out = h0

        with tf.name_scope("train"):
            h1 = h0
            for step in range(gibbs_sampling_steps):
                v1_prob = tf.nn.sigmoid(
                    tf.matmul(h1, tf.transpose(self.W)) + self.b_in)
                v1 = sample_prob(v1_prob)
                h1_prob = tf.nn.sigmoid(tf.matmul(v1, self.W) + self.b_out)
                h1 = sample_prob(h1_prob)

            # pozitivna faza
            w1_positive_grad = tf.matmul(tf.transpose(v0_prob), h0_prob)

            # negativna faza
            w1_negative_grad = tf.matmul(tf.transpose(v1_prob), h1_prob)

            dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(input_layer)[0])

            opt = tf.train.AdamOptimizer(1e-2)
            self.train_op = opt.apply_gradients([
                (-dw1, self.W),
                (-tf.reduce_mean(input_layer - v1_prob, 0), self.b_in),
                (-tf.reduce_mean(h0 - h1, 0), self.b_out)
            ])

        # rekonstrukcija ualznog vektora - koristimo vjerojatnost p(v=1)
        self.rec = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(self.W)) + self.b_in)
        err1 = input_layer - self.rec
        self.mse = tf.reduce_mean(err1 * err1)


def draw_weights(W, shape, N, interpolation="bilinear"):
    """Vizualizacija težina

    W -- vektori težina
    shape -- tuple dimenzije za 2D prikaz težina - obično dimenzije ulazne slike, npr. (28,28)
    N -- broj vektora težina
    """
    image = Image.fromarray(tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / 20)), 20),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)


def draw_reconstructions(ins, outs, states, shape_in, shape_state, Nh):
    """Vizualizacija ulaza i pripadajućih rekonstrkcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    plt.figure(figsize=(8, 12 * 4))
    for i in range(20):
        plt.subplot(20, 4, 4 * i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0,
                   vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.subplot(20, 4, 4 * i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in),
                   vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.subplot(20, 4, 4 * i + 3)
        plt.imshow(states[i][0:Nh].reshape(shape_state),
                   vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
    plt.tight_layout()
