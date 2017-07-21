import nltk
import numpy as np
from scipy.sparse import lil_matrix


class Datalist:

    def __init__(self, train=False):
        self.data = []
        self.train = train
        self.char_nums = Numberer()
        self.word_nums = Numberer()
        self.emo_nums = Numberer()
        self.max_len_char = -1
        self.max_len_word = -1

    def load(self, path):
        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')

                if line:
                    parts = line.split('\t')
                    chars = np.array([self.char_nums.number(c, self.train) for c in parts[1]], dtype=np.uint8)

                    if len(chars) > self.max_len_char:
                        self.max_len_char = len(chars)

                    words = np.array([self.word_nums.number(word, self.train) for word in nltk.word_tokenize(parts[1])], dtype=np.uint32)

                    if len(words) > self.max_len_word:
                        self.max_len_word = len(words)

                    emotion = self.emo_nums.number(parts[3], True)
                    self.data.append((chars, words, emotion))

    def create_coo_batches(self, batch_size):
        """
        Creates a list of lists of sparse char and word batches in COO format.

        :param batch_size: Size of individual batches
        :return: [char batches, word batches] in COO format
        """

        char_batch = lil_matrix((batch_size, self.max_len_char), dtype=np.uint8)
        word_batch = lil_matrix((batch_size, self.max_len_word), dtype=np.uint32)

        matrices = [[], []]
        for y, (chars, words, _) in enumerate(self):
            if y and not y % batch_size:
                self._sparse[0].append(char_batch.tocoo())
                self._sparse[1].append(word_batch.tocoo())
                char_batch = lil_matrix((batch_size, self.max_len_char), dtype=np.uint8)
                word_batch = lil_matrix((batch_size, self.max_len_word), dtype=np.uint32)

            for x, char_num in enumerate(chars): 
                char_batch[y % batch_size, x] = char_num

            for x, word_num in enumerate(words):
                word_batch[y % batch_size, x] = word_num

        return matrices

    def __iter__(self):
        for entry in self.data:
            yield entry

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "<Datalist with {} rows>".format(len(self.data))

    @property
    def n_chars(self):
        return self.char_nums.max()

    @property
    def n_words(self):
        return self.word_nums.max()

    @property
    def n_labels(self):
        return self.emo_nums.max()


class Numberer:

    def __init__(self):
        self.num2value = {}
        self.value2num = {}
        self.unkown_idx = 0
        self.idx = 1

    def number(self, value, train):

        if train and value not in self.value2num:
            self.value2num[value] = self.idx
            self.num2value[self.idx] = value
            self.idx += 1

        return self.value2num.get(value, self.unkown_idx)

    def max(self):
        return self.idx

    def value(self, num):
        return self.num2value.get(num, None)
