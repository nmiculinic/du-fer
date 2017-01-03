from model import VanilaRNN
from dataset import Dataset, data_root
import os
import numpy as np
import pickle


def oh(x, vocab_size):
    """
        x- > (batch_size, sequence_length)
        out -> (batch_size, sequence_length, vocab_size)
    """

    sol = np.zeros([*x.shape, vocab_size], dtype=np.float64)
    for idx in np.ndindex(*x.shape):
        sol[idx][x[idx]] = 1
    return sol


def sample(seed, n_sample, model):
    h = np.zeros((1, model.hidden_size))
    seed_oh = oh(seed, model.vocab_size)

    for char in seed_oh:
        # print(char.shape, char[np.newaxis, :].shape, h.shape)
        h, _ = model.rnn_step_forward(char[np.newaxis, :], h)

    sample = np.zeros((n_sample, ), dtype=np.int32)
    sample[:len(seed)] = seed
    for i in range(len(seed), n_sample):
        model_out = model.output(h[np.newaxis, :, :])
        sample[i] = np.random.choice(np.arange(model_out.shape[-1]), p=model_out.ravel())

        model_out[:] = 0
        model_out = model_out.reshape(1, -1)
        model_out[0, sample[i]] = 1
        h, _ = model.rnn_step_forward(model_out, h)

    return sample


def run_language_model(
    max_epochs,
    hidden_size=100,
    sequence_length=30,
    learning_rate=1e-1,
    sample_every=100
):
    ds = Dataset(
        batch_size=32,
        max_len=sequence_length
    )

    ds.preprocess(os.path.join(data_root, 'selected_conversations.txt'))
    vocab_size = ds.alphabet_size
    RNN = VanilaRNN(hidden_size, sequence_length, vocab_size, learning_rate)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((1, hidden_size))

    average_loss = 0

    save_path = "model_data"
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            batch, current_epoch, ds.batch_iter, RNN = pickle.load(f)
        print("===== Reading datamodel =====")

    while current_epoch < max_epochs:
        e, x, y = ds.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((1, hidden_size))
            # why do we reset the hidden state here?


        loss, h0 = RNN.step(h0, oh(x, vocab_size), oh(y, vocab_size))
        average_loss = 0.9 * average_loss + 0.1 * loss

        if batch % sample_every == 0:
            smp = sample(ds.encode("What is the meaning of life???\n\n"), 200, RNN)
            print("=====", average_loss, " EPOH", current_epoch, "BATCH", batch, "=====")
            print(ds.decode(smp))

        if batch % 1000 == 0:
            print("===== Saving datamodel =====")
            with open(save_path, "wb") as f:
                pickle.dump((batch, current_epoch, ds.batch_iter, RNN), f, protocol=4)
        batch += 1


if __name__ == "__main__":
    run_language_model(5)
    # x = np.random.randint(0, 3, size=(3, ))
    # print(x)
    # print(oh(x, 3))
    #
    #
    # x = np.random.randint(0, 3, size=(2, 3,))
    # print(x)
    # print(oh(x, 3))
