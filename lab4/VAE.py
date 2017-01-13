import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
import shutil

shutil.rmtree('train')

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

learning_rate = 0.001
batch_size = 100


n_hidden_recog_1 = 200  # 1 sloj enkodera
n_hidden_recog_2 = 200  # 2 sloj enkodera
n_hidden_gener_1 = 200  # 1 sloj dekodera
n_hidden_gener_2 = 200  # 2 sloj dekodera
n_z = 2  # broj skrivenih varijabli
n_input = 784  # MNIST data input (img shape: 28*28)


def draw_reconstructions(ins, outs, states, shape_in, shape_state):
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
        plt.imshow(states[i][0:(shape_state[0] * shape_state[1])
                             ].reshape(shape_state), interpolation="nearest")
        plt.colorbar()
        plt.title("States")
    plt.tight_layout()


def weight_variable(shape, name):
    """Kreiranje težina"""
    # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
    return tf.get_variable(name, shape=shape,
                           initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape):
    """Kreiranje pomaka"""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Prikupljanje podataka za Tensorboard"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def vae_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.softplus):
    """Kreiranje jednog skrivenog sloja"""
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(
                [input_dim, output_dim], layer_name + '/weights')
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
    return activations


tf.reset_default_graph()
with tf.Session() as sess:

    # definicije ulaznog tenzora
    x = tf.placeholder(tf.float32, [None, n_input])

    # TODO definirajte enkoiderski dio
    layer_e1 = vae_layer(x, n_input, n_hidden_recog_1, 'layer_e1')
    layer_e2 = vae_layer(layer_e1, n_hidden_recog_1,
                         n_hidden_recog_2, 'layer_e2')

    with tf.name_scope('z'):
        # definirajte skrivene varijable i pripadajući generator šuma
        z_mean = vae_layer(layer_e2, n_hidden_recog_2,
                           n_z, 'z_mean', act=tf.identity)
        z_log_sigma_sq = vae_layer(
            layer_e2, n_hidden_recog_2, n_z, 'z_log_sigma', act=tf.identity)
        eps = tf.random_normal((batch_size, n_z), 0, 1, dtype=tf.float32)

        z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 5)

        z = z_mean + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
        tf.summary.histogram('z/activations', z)

    # definirajte dekoderski dio
    layer_d1 = vae_layer(z, n_z, n_hidden_gener_1, 'layer_d1')
    layer_d2 = vae_layer(layer_d1, n_hidden_gener_1,
                         n_hidden_gener_2, 'layer_d2')

    # definirajte srednju vrijednost rekonstrukcije
    x_reconstr_mean = vae_layer(
        layer_d2, n_hidden_gener_2, n_input, 'layer_decoder')

    x_reconstr_mean_out = tf.nn.sigmoid(x_reconstr_mean)

    # definirajte dvije komponente funkcije cijene
    with tf.name_scope('costs'):
        # komponenta funkcije cijene - unakrsna entropija
        cost1_ce = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(x_reconstr_mean, x), axis=1)
        # komponenta funkcije cijene - KL divergencija
        cost_kl = 0.5 * (1 + z_log_sigma_sq - z_mean - tf.exp(z_log_sigma_sq))
        cost_kl = tf.reduce_sum(cost_kl, axis=1)

        cost_kl = tf.Print(cost_kl, [tf.shape(cost_kl)], message="bitches", first_n=5)
        cost = tf.reduce_mean(cost1_ce + cost_kl)   # average over batch
        # cost = tf.Print(cost, [tf.reduce_mean(cost1_ce), tf.reduce_mean], message="cost")
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('cost_CE', tf.reduce_mean(cost1_ce))
        tf.summary.scalar('cost_KL', tf.reduce_mean(cost_kl))

    # ADAM optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)

    # Prikupljanje podataka za Tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train/run1', sess.graph, flush_secs=5)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    n_epochs = 10

    for epoch in range(n_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            opt, cos, summ = sess.run((optimizer, cost, merged), feed_dict={x: batch_xs})
            avg_cost += cos / n_samples * batch_size
            train_writer.add_summary(summ, i + epoch * total_batch)

        if epoch % 10 == 9:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(avg_cost))
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs},
                                  options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'epoch%03d' % epoch)

            saver.save(sess, os.path.join('train', "model.ckpt"), epoch)

    train_writer.close()

    # vizualizacija rekonstrukcije i stanja
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct, z_out = sess.run(
        [x_reconstr_mean_out, z], feed_dict={x: x_sample})

    draw_reconstructions(x_sample, x_reconstruct, z_out, (28, 28), (4, 5))

    # Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli
    # - 1. način
    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = sess.run(z_mean, feed_dict={x: x_sample})

    plt.figure(figsize=(8, 6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.savefig('VAE_1.png')

    # Vizualizacija raspored testnih uzoraka u 2D prostoru skrivenih varijabli
    # - 2. način

    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28 * ny, 28 * nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]])
            x_mean = sess.run(x_reconstr_mean_out, feed_dict={
                              z: np.repeat(z_mu, 100, 0)})
            canvas[(nx - i - 1) * 28:(nx - i) * 28, j *
                   28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    plt.savefig('VAE_2.png')
