import numpy as np
import unittest
from unittest import skip
from sklearn.metrics import log_loss


def softmax(z):
    assert len(z.shape) == 3
    s = np.max(z, axis=2)
    s = s[:, :, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=2)
    div = div[:, :, np.newaxis]  # dito
    return e_x / div


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
        self.dW += np.dot(h_prev.T, dz) / grad_next.shape[0]
        self.dU += np.dot(x.T, dz) / grad_next.shape[0]
        self.db += np.sum(dz, axis=0) / grad_next.shape[0]

        return dh_prev

    def rnn_forward(self, x, h0, U=None, W=None, b=None):
        """
        Full unroll forward of the recurrent neural network with a
        hyperbolic tangent nonlinearity

        x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        h0 - initial hidden state (minibatch size x hidden size)
        U - input projection matrix (input dimension x hidden size)
        W - hidden to hidden projection matrix (hidden size x hidden size)
        b - bias of shape (hidden size x 1)
        """

        U = U if U is not None else self.U
        W = W if W is not None else self.W
        b = b if b is not None else self.b

        x = x.transpose(1, 0, 2)
        h = [h0]
        cache = []

        for x_item in x:
            h_item, cache_item = self.rnn_step_forward(x_item, h[-1], U, W, b)
            cache.append(cache_item)
            h.append(h_item)

        return np.array(h[1:]).transpose(1, 0, 2), cache  # Batch major h

    def rnn_backward(self, dh, cache):
        """
            dh -> (batch_size, sequence_length, hidden_size)
            cache -> (sequence_length, cahes)
        """
        self.init_backprop()
        assert dh.shape[1:] == (self.sequence_length, self.hidden_size)
        dh = dh.transpose(1, 0, 2)  # Switching to time major
        upstream_grad = np.zeros_like(dh[-1])
        for dh_item, cache_item in reversed(list(zip(dh, cache))):
            upstream_grad = self.rnn_step_backward(dh_item + upstream_grad, cache_item)

        return self.dU, self.dW, self.db

    def output(self, h, V=None, c=None):
        """
        Calculate the output probabilities of the network
        h - hidden states of the network for each timestep. the dimensionality of h is (batch size x sequence length x hidden
        V - the output projection matrix of dimension hidden size x vocabulary size
        c - the output bias of dimension vocabulary size x 1
        """

        if V is None:
            V = self.V
        if c is None:
            c = self.c

        logits = np.einsum('ijk,kl->ijl', h, V) + c[np.newaxis, :, :]
        return logits

    def output_loss_and_grads(self, h, y, V=None, c=None):
        """
        Calculate the loss of the network for each of the outputs

        h - hidden states of the network for each timestep.
         the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        y - the true class distribution - a tensor of dimension
         batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
         passing the argument. A fast way to create a one-hot vector from
         an id could be something like the following code:

          y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
          y[batch_id][timestep][batch_y[timestep]] = 1

         where y might be a list or a dictionary.

         V - the output projection matrix of dimension hidden size x vocabulary size
         c - the output bias of dimension vocabulary size x 1
        """

        if V is None:
            V = self.V
        if c is None:
            c = self.c

        batch_size = h.shape[0]
        np.testing.assert_array_equal(h.shape, (batch_size, self.sequence_length, self.hidden_size))

        o = self.output(h, V=V, c=c)

        np.testing.assert_array_equal(o.shape, (batch_size, self.sequence_length, self.vocab_size))

        yhat = softmax(o)
        loss = log_loss(y.reshape(-1, self.vocab_size), yhat.reshape(-1, self.vocab_size)) * self.sequence_length  # Since it computes average cross_entropy loss, not accounting for sequence_length
        do = yhat - y  # (batch_size, sequence_length, vocab_size)
        assert do.shape == (batch_size, self.sequence_length, self.vocab_size)

        dV = np.zeros_like(V)
        dc = np.zeros_like(c)
        dh = []
        for ddo, hh in zip(do.transpose(1, 0, 2), h.transpose(1, 0, 2)):
            assert ddo.shape == (batch_size, self.vocab_size)
            assert hh.shape == (batch_size, self.hidden_size)
            dV += np.dot(hh.T, ddo) / batch_size
            dc += np.average(ddo, axis=0)
            dh.append(np.dot(ddo, V.T))
        dh = np.array(dh).transpose(1, 0, 2)  # batch major
        assert dh.shape == h.shape
        assert dh.shape == (batch_size, self.sequence_length, self.hidden_size)

        self.dV = dV
        self.dc = dc

        return loss, dh

    def apply_grad(self, eps=1e-6):
        self.memory_U += np.square(self.dU)
        self.memory_W += np.square(self.dW)
        self.memory_b += np.square(self.db)
        self.memory_V += np.square(self.dV)
        self.memory_c += np.square(self.dc)

        self.U -= self.learning_rate * self.dU / np.sqrt(self.memory_U + eps)
        self.W -= self.learning_rate * self.dW / np.sqrt(self.memory_W + eps)
        self.b -= self.learning_rate * self.db / np.sqrt(self.memory_b + eps)
        self.V -= self.learning_rate * self.dV / np.sqrt(self.memory_V + eps)
        self.c -= self.learning_rate * self.dc / np.sqrt(self.memory_c + eps)


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
            15,
            vocab_size,
            0.1)

        x = np.zeros([1, vocab_size])
        x[:, 1] = 1
        self.x = x

        x4 = np.zeros([4, vocab_size])
        # x4[[0, 1], 1] = 1
        # x4[[2, 3], 0] = 1
        x4[:, 1] = 1
        self.x4 = x4

        self.h0 = np.array([[1.1, 2.1, 0.1]])

        self.x_long = np.array([self.x4 for _ in range(self.rnn.sequence_length)]).transpose(1, 0, 2)

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
            return np.average(np.sum(h, axis=1))  # Fictional loss = sum(h)

        np.testing.assert_allclose(
            np.average(d_prev, axis=0).reshape(1, 3),  # Sum of all gradients
            num_grad(self.h0, fn)
        )

    def test_backprob_W(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(w):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, W=w)  # Fictional loss = sum(h)
            return np.average(np.sum(h, axis=1))

        np.testing.assert_allclose(
            self.rnn.dW,  # average of all gradients
            num_grad(self.rnn.W, fn)
        )

    def test_backprob_U(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(u):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, U=u)  # Fictional loss = sum(h)
            return np.average(np.sum(h, axis=1))

        np.testing.assert_allclose(
            self.rnn.dU,  # Average of all gradients
            num_grad(self.rnn.U, fn)
        )

    def test_backprob_b(self):
        h, cache = self.rnn.rnn_step_forward(self.x4, self.h0)
        self.rnn.init_backprop()
        self.rnn.rnn_step_backward(np.ones([4, 3]), cache)

        def fn(b):
            h, _ = self.rnn.rnn_step_forward(self.x4, self.h0, b=b)  # Fictional loss = sum(h)
            return np.average(np.sum(h, axis=1))

        np.testing.assert_allclose(
            self.rnn.db,  # Average of all gradients
            num_grad(self.rnn.b, fn)
        )

    def test_backprop_len(self):
        self.rnn.init_backprop()
        h, cache = self.rnn.rnn_forward(self.x_long, self.h0)
        dh = np.array([i * np.ones([self.x_long.shape[0], self.rnn.hidden_size]) for i in range(1, self.rnn.sequence_length + 1)]).transpose(1, 0, 2)

        self.rnn.rnn_backward(dh, cache)

        def fn(w):
            h, _ = self.rnn.rnn_forward(self.x_long, self.h0, W=w)
            sol = 0
            h = h.transpose(1, 0, 2)  # switch to time major
            for i, hh in enumerate(h):
                sol += np.average(np.sum((i + 1) * hh, axis=1))
            return sol

        np.testing.assert_allclose(
            self.rnn.dW,  # Sum of all gradients
            num_grad(self.rnn.W, fn)
        )

    def test_output_dv(self):
        h, cache = self.rnn.rnn_forward(self.x_long, self.h0)
        loss, dh = self.rnn.output_loss_and_grads(h, self.x_long)

        def fn(v):
            loss, _ = self.rnn.output_loss_and_grads(h, self.x_long, V=v)
            return loss

        np.testing.assert_allclose(
            self.rnn.dV,  # Sum of all gradients
            num_grad(self.rnn.V, fn),
            rtol=1e-3
        )

    def test_output_dc(self):
        h, cache = self.rnn.rnn_forward(self.x_long, self.h0)
        loss, dh = self.rnn.output_loss_and_grads(h, self.x_long)

        def fn(c):
            loss, _ = self.rnn.output_loss_and_grads(h, self.x_long, c=c)
            return loss

        np.testing.assert_allclose(
            self.rnn.dc,  # Sum of all gradients
            num_grad(self.rnn.c, fn),
            rtol=1e-3
        )


if __name__ == '__main__':
    unittest.main()
