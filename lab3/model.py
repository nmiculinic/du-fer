import numpy as np
import unittest
from unittest import skip


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

        self.U = np.random.normal(size=[vocab_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... input projection
        self.W = np.random.normal(size=[hidden_size, hidden_size], scale=1.0 / np.sqrt(hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros([1, hidden_size])

        self.V = np.random.normal(size=[hidden_size, vocab_size], scale=1.0 / np.sqrt(vocab_size))  # ... output projection
        self.c = np.zeros([1, vocab_size])  # ... output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U=None, W=None, b=None):
        """
        A single time step forward of a recurrent neural network with a
        hyperbolic tangent nonlinearity.

        x - input data (minibatch size x input dimension)
        h_prev - previous hidden state (minibatch size x hidden size) or (1xhidden_size) with broadcasting
        U - input projection matrix (input dimension x hidden size)
        W - hidden to hidden projection matrix (hidden size x hidden size)
        b - bias of shape (hidden size x 1)
        """

        U = U if U is not None else self.U
        W = W if W is not None else self.W
        b = b if b is not None else self.b
        if h_prev.shape[0] == 1:
            h_prev = np.broadcast_to(h_prev, (x.shape[0], h_prev.shape[1]))
        assert x.shape[0] == h_prev.shape[0]

        h = np.dot(x, U) + np.dot(h_prev, W) + b
        h = np.tanh(h)
        # print("__54", h_prev.shape, h.shape, x.shape)
        return h, (h, h_prev, x)

    def init_backprop(self):
        self.dU = np.zeros_like(self.U)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def rnn_step_backward(self, grad_next, cache):
        """
        A single time step backward of a recurrent neural network with a
        hyperbolic tangent nonlinearity.

        grad_next - (N, out) upstream gradient of the loss with respect to the next hidden state and current output
        cache - cached information from the forward pass
        """

        th, h_prev, x = cache
        dz = grad_next * (1 - th**2)
        dh_prev = np.dot(dz, self.W.T)
        # print("__73__", h_prev.shape, dz.shape)
        self.dW += np.dot(h_prev.T, dz)
        self.dU += np.dot(x.T, dz)
        self.db += np.sum(dz, axis=0)

        # print(grad_next.shape, th.shape, dz.shape, dh_prev.shape)
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


def num_grad(var, fn, eps=1e-7):
    grad = np.zeros_like(var)
    for idx in np.ndindex(*var.shape):
        init = np.copy(var)
        init[idx] += eps
        up = fn(init)
        init = np.copy(var)
        init[idx] -= eps
        down = fn(init)
        grad[idx] = (up - down) / (2 * eps)
        # print(idx, up - down)

    return grad


class TestModel(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)

        vocab_size = 2

        self.rnn = VanilaRNN(
            3,
            5,
            vocab_size,
            0.1)

        x = np.zeros([1, vocab_size])
        x[np.arange(x.shape[0]), [1]] = 1
        self.x = x

        x4 = np.zeros([4, vocab_size])
        x4[[0, 1], 1] = 1
        x4[[2, 3], 0] = 1
        self.x4 = x4

        self.h0 = np.array([[1.1, 2.1, 0.1]])

    @skip("debug only")
    def test_print(self):
        print("==== PRINT VARS ====")
        print("X")
        print(self.x)
        print("X4")
        print(self.x4)
        print("h0")
        print(self.h0)

    # def test_forward_prop(self):
    #     h, _ = self.rnn.rnn_step_forward(self.x, self.h0)
    #     # print(h)
    #     np.testing.assert_allclose(h, [[0.98763894, -0.20664687, -0.76075404]])

    def test_num_grad(self):
        grad = num_grad(np.array(2.0), lambda x: x**3)
        np.testing.assert_allclose(12.0, grad)

    def test_num_grad2(self):
        grad = num_grad(np.array([2.0, 3.0]), lambda x: np.sum(x**3))
        np.testing.assert_allclose([12.0, 27.0], grad)

    def test_backprob_h(self):

        self.rnn.init_backprop()
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        d_prev = self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(h0):
            h, _ = self.rnn.rnn_step_forward(self.x4, h0)
            return np.sum(h)  # Fictional loss = sum(h)

        np.testing.assert_allclose(
            np.sum(d_prev, axis=0).reshape(1, 3),  # Sum of all gradients
            num_grad(self.h0, fn)
        )

    def test_backprob_W(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(w):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, W=w)  # Fictional loss = sum(h)
            return np.sum(h)

        np.testing.assert_allclose(
            self.rnn.dW,  # Sum of all gradients
            num_grad(self.rnn.W, fn)
        )

    def test_backprob_U(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(u):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, U=u)  # Fictional loss = sum(h)
            return np.sum(h)

        np.testing.assert_allclose(
            self.rnn.dU,  # Sum of all gradients
            num_grad(self.rnn.U, fn)
        )

    def test_backprob_b(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(b):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, b=b)  # Fictional loss = sum(h)
            return np.sum(h)

        np.testing.assert_allclose(
            self.rnn.db,  # Sum of all gradients
            num_grad(self.rnn.b, fn)
        )

if __name__ == '__main__':
    unittest.main()
