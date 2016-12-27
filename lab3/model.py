import numpy as np





def rnn_step_backward(self, grad_next, cache):
    # A single time step backward of a recurrent neural network with a
    # hyperbolic tangent nonlinearity.

    # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
    # cache - cached information from the forward pass

    dh_prev, dU, dW, db = None, None, None, None

    # compute and return gradients with respect to each parameter
    # HINT: you can use the chain rule to compute the derivative of the
    # hyperbolic tangent function and use it to compute the gradient
    # with respect to the remaining parameters

    return dh_prev, dU, dW, db


def rnn_backward(self, dh, cache):
    # Full unroll forward of the recurrent neural network with a
    # hyperbolic tangent nonlinearity

    dU, dW, db = None, None, None

    # compute and return gradients with respect to each parameter
    # for the whole time series.
    # Why are we not computing the gradient with respect to inputs (x)?

    return dU, dW, db


class VanilaRNN():
    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal([vocab_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... input projection
        self.W = np.random.normal([hidden_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros([hidden_size])

        self.V = np.random.normal([hidden_size, vocab_size], scale=1.0 / np.sqrt(vocab_size))  # ... output projection
        self.c = np.zeros([vocab_size])  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        """
        A single time step forward of a recurrent neural network with a
        hyperbolic tangent nonlinearity.

        x - input data (minibatch size x input dimension)
        h_prev - previous hidden state (minibatch size x hidden size)
        U - input projection matrix (input dimension x hidden size)
        W - hidden to hidden projection matrix (hidden size x hidden size)
        b - bias of shape (hidden size x 1)
        """

        h = np.dot(x, U) + np.dot(h_prev, W) + b
        h = np.tanh(h)

        return h, h

    def init_backprop(self):
        self.dU = np.zeros_like(self.U)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def rnn_step_backward(self, grad_next, cache):
        """
        A single time step backward of a recurrent neural network with a
        hyperbolic tangent nonlinearity.

        grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        cache - cached information from the forward pass
        """

        th = cache
        dz = np.dot(grad_next, 1 - th**2)
        dh_prev = np.dot(dz, self.W)
        self.dW = np.dot(dz.T, dh_prev)

        return dh_prev

    def rnn_forward(self, x, h0, U, W, b):
        """
        Full unroll forward of the recurrent neural network with a
        hyperbolic tangent nonlinearity

        x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        h0 - initial hidden state (minibatch size x hidden size)
        U - input projection matrix (input dimension x hidden size)
        W - hidden to hidden projection matrix (hidden size x hidden size)
        b - bias of shape (hidden size x 1)
        """

        x = x.transpose(1, 0, 2)
        h = [h0]
        cache = []

        for x_item in x:
            h_item, cache_item = self.rnn_step_forward(x_item, h[-1], U, W, b)
            cache.append(cache_item)
            h.append(h_item)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, cache


    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU, dW, db = None, None, None

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db

if __name__ == "__main__":
    rnn = VanilaRNN(2, 5, 3, 0.1)

    x = np.zeros([2, 3])
    x[:, [1,2]] = 1
    print(x)

    # rnn.rnn_step_forward([1,2,3], , U, W, b)
    rnn.init_backprop()
