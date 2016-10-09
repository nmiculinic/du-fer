import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


class TFLogreg:

    def __init__(self, D, C, param_delta=0.5, l2=0):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        self.X = tf.placeholder(tf.float32, [None, D], "X")
        self.Yoh = tf.placeholder(tf.float32, [None, C], "Yp")

        self.w = tf.Variable(tf.zeros([D, C]))
        self.b = tf.Variable(tf.zeros([C]))

        self.logits = tf.matmul(self.X, self.w) + self.b
        self.yp = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.logits, self.Yoh)) + l2 * tf.nn.l2_loss(self.w)
        self.trainer = tf.train.GradientDescentOptimizer(param_delta)
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

    model = LogisticRegression(C=1e5)
    model.fit(X, Y)
    data.graph_data_pred(X, Y, model)
    plt.savefig("lab1/logreg.png")

    for ll in np.logspace(-6, 4, num=50):

        print("lambda", ll)
        # izgradi graf:
        tflr = TFLogreg(X.shape[1], Yoh.shape[1], 0.5, ll)

        # nauči parametre:
        tflr.train(X, Yoh, 1000)
        # dohvati vjerojatnosti na skupu za učenje
        probs = tflr.eval(X)
        ypp = np.argmax(probs, axis=1)
        print(classification_report(Y.reshape(-1), ypp))
        cm = confusion_matrix(Y.reshape(-1), ypp)
        print("Confusion matrix\n", cm)
        # iscrtaj rezultate, decizijsku plohu

        data.graph_data_pred(X, Y, tflr)
        plt.savefig('lab1/lr_%f.png' % ll)
        tflr.sess.close()
