import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import data
from sklearn.preprocessing import OneHotEncoder


class TFDeep:
    def __init__(self, layers, param_delta=0.5, l2=0):

        self.X = tf.placeholder(tf.float32, [None, layers[0]], "X")
        self.Yoh = tf.placeholder(tf.float32, [None, layers[-1]], "Yp")

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

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        for i in range(param_niter):
                self.sess.run(self.train_op, feed_dict={
                              self.X: X,
                              self.Yoh: Yoh_})

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        return self.sess.run(self.yp, feed_dict={self.X: X})

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
    tflr = tfdeep([d, 10, c], 0.001, ll)
    tflr.count_params()

    # nauči parametre:
    tflr.train(x, yoh, 10000)
    # dohvati vjerojatnosti na skupu za učenje
    probs = tflr.eval(x)
    ypp = np.argmax(probs, axis=1)
    print(classification_report(y.reshape(-1), ypp))
    cm = confusion_matrix(y.reshape(-1), ypp)
    print("confusion matrix\n", cm)
    # iscrtaj rezultate, decizijsku plohu

    data.graph_data_pred(X, Y, tflr)
    plt.show()
    tflr.sess.close()
