import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import seaborn as sns
from RBM import draw_reconstructions, draw_weights, RBM
sns.set_style("dark")

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels
Nv = 784
v_shape = (28, 28)
Nh = 100
h1_shape = (10, 10)


batch_size = 100
epochs = 20
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

X = tf.placeholder("float", [None, 784])

rbm = RBM(X, Nh)
rbm.create_rec()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(total_batch):
        batch, label = mnist.train.next_batch(batch_size)
        err, _ = sess.run([rbm.mse, rbm.train_op], feed_dict={X: batch})

        if i % 100 == 0:
            print(i, "[%d]" % total_batch, err)

    w1s = rbm.W.eval()
    vb1s = rbm.b_in.eval()
    hb1s = rbm.b_out.eval()
    vr, h1s = sess.run([rbm.rec, rbm.out], feed_dict={X: teX[:20, :]})

# vizualizacija težina
draw_weights(w1s, v_shape, Nh)
plt.savefig("ex1_RBM_weights")

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr, h1s, v_shape, h1_shape, Nh)
plt.savefig("ex1_RBM_reconstruction")
