import nltk
import numpy as np
from scipy.sparse import lil_matrix


class Datalist:

    def __init__(self, trained_numberers=None):
        """
        Creates a new Datalist object.
        if no pretrained numberers are given new numbereres are trained while loading.

        :param trained_numberers: A tuple of (char_nums, word_nums, emo_nums) or None
        """
        self.data = []
        self.train = trained_numberers is None
        if self.train:
            self.char_nums = Numberer()
            self.word_nums = Numberer()
            self.emo_nums = Numberer()
        else:
            self.char_nums, self.word_nums, self.emo_nums = trained_numberers

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

    def create_coo_batches(self, batch_size, mode='indices'):
        """
        Creates a list of lists of sparse char and word batches in COO format
        together with the lengths of each char and word sequence and labels for each sample.

        :param batch_size: Size of individual batches
        :param mode: Type of the batch: 'indices' or 'multi_hot'
        :return: [char batches, word batches],
            [char sequence length batches, word sequence length batches],
            label batches
        """
        if mode not in {'indices', 'multi_hot'}:
            raise ValueError("Batch mode must be one of ('indices', 'multi_hot')")

        if mode == 'indices':
            char_width = self.max_len_char
            word_width = self.max_len_word
        else:
            char_width = self.n_chars
            word_width = self.n_words

        char_batch = lil_matrix((batch_size, char_width), dtype=np.uint8)
        word_batch = lil_matrix((batch_size, word_width), dtype=np.uint32)
        chars_seq_length = np.empty(batch_size, dtype=np.uint32)
        words_seq_length = np.empty(batch_size, dtype=np.uint32)
        label_batch = np.empty(batch_size, dtype=np.uint8)

        matrices = [[], []]
        seq_length_batches = [[], []]
        labels = []

        for y, (chars, words, emotion) in enumerate(self):
            if y and not y % batch_size:
                #append previous batch data
                matrices[0].append(char_batch.tocoo())
                matrices[1].append(word_batch.tocoo())
                seq_length_batches[0].append(chars_seq_length)
                seq_length_batches[1].append(words_seq_length)
                labels.append(label_batch)

                #create empty representations for the next batch
                char_batch = lil_matrix((batch_size, char_width), dtype=np.uint8)
                word_batch = lil_matrix((batch_size, word_width), dtype=np.uint32)
                char_seq_length = np.empty(batch_size)
                word_seq_length = np.empty(batch_size)
                label_batch = np.empty(batch_size)

            chars_length = len(chars)
            words_length = len(words)
            
            index = y % batch_size
            if mode == 'indices':
                char_batch[index, :chars_length] = chars
                word_batch[index, :words_length] = words
            else:
                char_batch[index, chars] = 1
                word_batch[index, words] = 1

            chars_seq_length[index] = chars_length
            words_seq_length[index] = words_length
            label_batch[index] = emotion

        return matrices, seq_length_batches, labels

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
    def numberers(self):
        return self.char_nums, self.word_nums, self.emo_nums

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
