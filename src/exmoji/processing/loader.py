import nltk
import numpy as np


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

    def __iter__(self):
        for entry in self.data:
            yield entry

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

        if value not in self.value2num and train:
            self.value2num[value] = self.idx
            self.num2value[self.idx] = value
            self.idx += 1

        return self.value2num.get(value, self.unkown_idx)

    def max(self):
        return self.idx

    def value(self, num):
        return self.num2value.get(num, None)
