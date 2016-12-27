import os
import numpy as np

data_root = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '../fer/code/lab3/data'))


class Dataset():

    def __init__(self, batch_size=None, max_len=None):
        self.batch_size = batch_size or 32
        self.max_len = max_len or 20
        self.batch_iter = 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        data = np.array([ord(x) for x in data], dtype=np.int32)

        bincnt = np.bincount(data)
        arg_srt = np.array(list(reversed(np.argsort(bincnt))))

        self.char2id = np.zeros_like(bincnt)
        self.id2char = np.zeros_like(bincnt)
        for i, elm in enumerate(arg_srt):
            if bincnt[elm] > 0:
                # print(bincnt[elm], chr(elm))
                self.char2id[elm] = i
                self.id2char[i] = elm
            else:
                self.char2id[elm] = 0
                self.id2char[i] = ord(' ')

        self.alphabet_size = np.max(self.char2id) + 1

        self.data = self.char2id[data]
        self.x = self.data[:-1]
        self.y = self.data[1:]

        # print(self.x[:10], self.y[:10], self.alphabet_size, np.max(self.char2id), np.max(self.id2char))

    def encode(self, sequence):
        return self.char2id[[ord(x) for x in sequence]]

    def decode(self, encoded_sequence):
        return "".join([chr(x) for x in self.id2char[encoded_sequence]])

    def next_minibatch(self):
        new_epoch = False
        if self.batch_size * self.max_len + self.batch_iter > self.x.shape[0]:
            self.batch_iter = 0
            new_epoch = True

        a = self.batch_iter
        b = a + self.batch_size * self.max_len

        batch_x, batch_y = self.x[a:b], self.y[a:b]

        return new_epoch, batch_x.reshape(self.batch_size, self.max_len), batch_y.reshape(self.batch_size, self.max_len)


ds = Dataset()
ds.preprocess(os.path.join(data_root, 'selected_conversations.txt'))

if __name__ == "__main__":
    print(
        ds.decode(ds.next_minibatch()[1][0]),
        '\n',
        ds.decode(ds.next_minibatch()[2][0]),
    )
