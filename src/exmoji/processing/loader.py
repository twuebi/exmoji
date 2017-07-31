import nltk
import numpy as np


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
        self.max_len_sentences = -1

    def load(self, path):
        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')

                if line:
                    parts = line.split('\t')

                    sents = nltk.sent_tokenize(parts[1], language="german")
                    if len(sents) > self.max_len_sentences:
                        self.max_len_sentences = len(sents)

                    chars = [[self.char_nums.number(c, self.train) for c in sent] for sent in sents]

                    max_sentence_chars = max(map(len, chars))
                    if max_sentence_chars > self.max_len_char:
                        self.max_len_char = max_sentence_chars

                    sentences = [
                        [self.word_nums.number(word, self.train) for word in nltk.word_tokenize(sent, language="german")]
                        for sent in sents
                    ]

                    max_sentence_len = max(map(len, sentences))
                    if max_sentence_len > self.max_len_word:
                        self.max_len_word = max_sentence_len

                    emotion = self.emo_nums.number(parts[3], True)
                    self.data.append((chars, sentences, emotion))

    def create_batches(self, batch_size, mode='indices'):
        if mode not in {'indices', 'multi_hot'}:
            raise ValueError("Batch mode must be one of ('indices', 'multi_hot')")

        if mode == 'indices':
            char_width = self.max_len_char
            word_width = self.max_len_word
        else:
            char_width = self.n_chars
            word_width = self.n_words

        char_batch = np.zeros((batch_size, self.max_len_sentences, char_width), dtype=np.uint8)
        word_batch = np.zeros((batch_size, self.max_len_sentences, word_width), dtype=np.uint32)
        sentence_lengths = np.empty(batch_size, dtype=np.uint32)
        chars_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
        words_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
        label_batch = np.empty(batch_size, dtype=np.uint8)

        matrices = [[], []]
        seq_length_batches = [[], [], []]
        labels = []

        for y, (chars, words, emotion) in enumerate(self):
            if y and not y % batch_size:
                #append previous batch data
                matrices[0].append(char_batch)
                matrices[1].append(word_batch)
                seq_length_batches[0].append(chars_seq_length)
                seq_length_batches[1].append(words_seq_length)
                seq_length_batches[2].append(sentence_lengths)
                labels.append(label_batch)

                #create empty representations for the next batch
                char_batch = np.zeros((batch_size, self.max_len_sentences, char_width), dtype=np.uint8)
                word_batch = np.zeros((batch_size, self.max_len_sentences, word_width), dtype=np.uint32)
                sentence_lengths = np.empty(batch_size, dtype=np.uint32)
                chars_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
                words_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
                label_batch = np.empty(batch_size, dtype=np.uint8)

            chars_lengths = list(map(len, chars))
            words_lengths = list(map(len, words))
            sentence_length = len(chars)
            
            index = y % batch_size
            for i, (char, word, char_len, word_len) in enumerate(zip(chars, words, chars_lengths, words_lengths)):
                if mode == 'indices':
                    char_batch[index, i, :char_len] = char
                    word_batch[index, i, :word_len] = word
                else:
                    char_batch[index, i, char] = 1
                    word_batch[index, i, word] = 1

            sentence_lengths[index] = sentence_length
            chars_seq_length[index, :sentence_length] = chars_lengths
            words_seq_length[index, :sentence_length] = words_lengths
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
