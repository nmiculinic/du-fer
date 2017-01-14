import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import seaborn as sns
from RBM import draw_reconstructions, draw_weights, RBM, sample_prob
sns.set_style("dark")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels
Nv = 784
v_shape = (28, 28)
Nh = 100
h1_shape = (10, 10)


batch_size = 100
epochs = 5
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

X = tf.placeholder("float", [None, 784])

with tf.variable_scope("l1"):
    rbm1 = RBM(X, 100)
with tf.variable_scope("l2"):
    rbm2 = RBM(rbm1.out_prob, 784, inbias=rbm1.b_out)

rbm2.create_rec()
rbm1.create_rec(rbm2.rec)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(rbm2.W, tf.transpose(rbm1.W)))

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([rbm1.mse, rbm1.train_op], feed_dict={X: batch})

        if i % 100 == 0:
            print("RBM1", i, "[%d]" % total_batch, err)

    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([rbm2.mse, rbm2.train_op], feed_dict={X: batch})

        if i % 100 == 0:
            print("RBM2", i, "[%d]" % total_batch, err)

    w1s = rbm1.W.eval()
    w2s = rbm2.W.eval()
    vb1s = rbm1.b_in.eval()
    hb1s = rbm1.b_out.eval()
    vr, h1s, h2s = sess.run([rbm1.rec, rbm1.out, rbm2.out], feed_dict={X: teX[:20, :]})

# vizualizacija težina
draw_weights(w1s, v_shape, Nh)
plt.savefig("ex2b_RBM_weights_1")
draw_weights(w2s, (10, 10), 784)
plt.savefig("ex2b_RBM_weights_2")

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape, Nh)
plt.savefig("ex2b_RBM_reconstruction_1")

draw_reconstructions(teX, vr, h2s, v_shape, (28, 28), 784)
plt.savefig("ex2b_RBM_reconstruction_2")
