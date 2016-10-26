import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from time import strftime

from os import path
logdir = path.join(path.dirname(__file__), 'logs')
print(logdir)


class TFDeep:
    def __init__(self, layers, param_delta=0.001, l2=0, ldir=strftime("%d_%b_%Y_%H:%M:%S")):

        self.X = tf.placeholder(tf.float32, [None, layers[0]], "X_input")
        self.Yoh = tf.placeholder(tf.float32, [None, layers[-1]], "Yp_target")

        net = self.X

        for k in range(len(layers) - 1):
            i = layers[k]
            j = layers[k + 1]
            w = tf.Variable(tf.random_normal([i, j], 0, (2 / i)**0.5))  # Xavier initialization for Relu
            b = tf.Variable(tf.constant(0, tf.float32, [j]))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, l2 * tf.nn.l2_loss(w))
            net = tf.matmul(net, w) + b
            if k + 1 < len(layers):
                net = tf.nn.relu(net)

        self.logits = net
        self.yp = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits, self.Yoh)) + np.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.trainer = tf.train.AdamOptimizer(param_delta)
        self.train_op = self.trainer.minimize(self.loss)
        tf.scalar_summary('loss', self.loss)
        self.sess = tf.Session()

        correct_prediction = tf.equal(tf.argmax(self.yp, 1), tf.argmax(self.Yoh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.scalar_summary("accuracy", accuracy)

        self.merged = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter(path.join(logdir, ldir, 'train'), self.sess.graph)

        self.val_writer = tf.train.SummaryWriter(path.join(logdir, ldir, 'val'),
                                      self.sess.graph)


        self.sess.run(tf.initialize_all_variables())

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        self.transform = StandardScaler()
        X = self.transform.fit_transform(X)

        for i in range(param_niter):
            if i % 100 == 0 or i == param_niter - 1:
                summary, _ = self.sess.run([self.merged, self.train_op], feed_dict={
                              self.X: X,
                              self.Yoh: Yoh_})

                self.train_writer.add_summary(summary, i)
            else:
                self.sess.run([self.train_op], feed_dict={
                              self.X: X,
                              self.Yoh: Yoh_})


    def fit(self, X, Y, Xv, Yv, batch_size, param_niter):


        self.transform = StandardScaler()
        X = self.transform.fit_transform(X)
        N = X.shape[0]

        for i in range(param_niter):
            perm = np.random.permutation(N)
            for idx in range(batch_size, N + 1, batch_size):
                idxs = perm[idx - batch_size:idx]
                batch_xs = X[idxs]
                batch_ys = Y[idxs]
                print(batch_xs.shape)
                self.sess.run(self.train_op, feed_dict={self.X: batch_xs, self.Yoh: batch_ys})

            if i % 100 == 0 or i == param_niter - 1:
                summary = self.sess.run(self.merged, feed_dict={
                              self.X: X,
                              self.Yoh: Y})

                self.train_writer.add_summary(summary, i)

                summary = self.sess.run(self.merged, feed_dict={
                              self.X: Xv,
                              self.Yoh: Yv})

                self.val_writer.add_summary(summary, i)

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        return self.sess.run(self.yp, feed_dict={self.X: self.transform.transform(X)})

    def predict(self, X):
        return np.argmax(self.eval(X), axis=1)

    def count_params(self):
        for var in tf.trainable_variables():
            print("var", var.name)

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh
    D = 2
    C = 2
    X, Y = data.sample_gmm_2d(5, C, 10)

    oh = OneHotEncoder(sparse=False)
    oh.fit(Y)
    Yoh = oh.transform(Y)

    Yoh.shape
    X.shape

    ll = 0
    print("lambda", ll)
    # izgradi graf:
    tflr = TFDeep([D, 10, 10, C], 0.001, ll)
    tflr.count_params()

    # nauči parametre:
    tflr.train(X, Yoh, 10000)
    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(X)
    ypp = np.argmax(probs, axis=1)
    print(classification_report(Y.reshape(-1), ypp))
    cm = confusion_matrix(Y.reshape(-1), ypp)
    print("confusion matrix\n", cm)
    # iscrtaj rezultate, decizijsku plohu

    data.graph_data_pred(X, Y, tflr)
    # plt.show()
    tflr.sess.close()
