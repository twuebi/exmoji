from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pdb

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2


class SentenceModel:
    def __init__(self,
                 batch_size,
                 sentences_size,
                 w_input_size,
                 label_size,
                 n_words_per_sent,
                 n_sents_per_doc,
                 n_words,
                 phase):

        self._sent_in = tf.placeholder(dtype=tf.int32, name='sentences_in', shape=[batch_size,n_sents_per_doc,n_words_per_sent])
        print(self._sent_in)
        inp = tf.reshape(self._sent_in,[batch_size,-1])
        print(inp)
        self._words = tf.get_variable('words', initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[n_words, 100], dtype=tf.float32)
        reg = tf.nn.l2_loss(self._words)
        self._w_lens = tf.placeholder(tf.int32)
        self._s_lens = tf.placeholder(tf.int32, [batch_size])
        print(self._w_lens)
        print(self._s_lens)

        word_lookup = tf.nn.embedding_lookup(self._words, inp)
        if phase == Phase.Train:
            word_lookup = tf.nn.dropout(word_lookup,0.85)
        print(word_lookup)
        word_lookup = tf.reshape(word_lookup,[batch_size,-1,100])
        print(word_lookup)
        seq = tf.reshape(tf.reduce_sum(self._w_lens[:,:n_sents_per_doc],axis=1),[batch_size])
        print(seq)

        fw_cell = rnn.LSTMCell(64,cell_clip=5 ,initializer=tf.contrib.layers.xavier_initializer())

        bw_cell = rnn.LSTMCell(64,cell_clip=5, initializer=tf.contrib.layers.xavier_initializer())
        if phase == Phase.Train:
            fw_cell = rnn.DropoutWrapper(fw_cell,0.5)
            bw_cell = rnn.DropoutWrapper(bw_cell,0.5)

        _, rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,word_lookup,sequence_length=seq,
                                                     dtype=tf.float32, swap_memory=True)
        hidden = tf.concat([rnn_out[0].h,rnn_out[1].h],axis=1)
        print(hidden)
        w = tf.get_variable("w", initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[hidden.shape[1], label_size], dtype=tf.float32)
        b = tf.get_variable("bias", shape=[1], dtype=tf.float32)
        logits = tf.matmul(hidden, w) + b

        if phase == Phase.Validation or phase == Phase.Train:
            self._y = tf.placeholder(tf.int32, shape=[batch_size], name='lbl')
            # print(label_size)
            self._labels = labels = tf.one_hot(self._y,label_size)
            #losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=logits)
            self._losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._y,
                                                                    logits=logits)
            self._loss = loss = tf.reduce_sum(self._losses)
            self._predictions = res = tf.argmax(logits, axis=1)
        if phase == Phase.Train:
            self._train_op = tf.train.AdamOptimizer().minimize(loss+reg*0.1)

        if phase == Phase.Validation:
            print(labels)
            print(res)
            # get highest scoring classes for all outputs, returns booleans
            correct = tf.equal(res, tf.argmax(labels, axis=1))
            print(correct)
            # get average of the booleans == accuracy
            self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    @property
    def labels(self):
        return self._labels

    @property
    def losses(self):
        return self._losses

    @property
    def predictions(self):
        return self._predictions

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens_s(self):
        return self._s_lens

    # @property
    # def lens_c(self):
    #     return self._c_lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def char_in(self):
        return self._char_in

    @property
    def sent_in(self):
        return self._sent_in

    @property
    def lens_w(self):
        return self._w_lens

    @property
    def y(self):
        return self._y

class Model:
    def __init__(self,
                 batch_size,
                 w_input_size,
                 c_input_size,
                 c_length,
                 w_length,
                 labels,
                 n_chars,
                 n_words,
                 phase):
        label_size = labels.shape[2]

        self._char_in = tf.placeholder(dtype=tf.int32, name='char_in', shape=[batch_size, c_input_size])
        self._word_in = tf.placeholder(name='words_in', dtype=tf.int32, shape=[batch_size, w_input_size])

        self._words = tf.get_variable('words', initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[n_words, 15], dtype=tf.float16)

        # self._chars = tf.get_variable('chars', initializer=tf.contrib.layers.xavier_initializer(),
        #                               shape=[n_chars, 20], dtype=tf.float32)

        self._w_lens = tf.placeholder(tf.int32, [batch_size])
        # self._c_lens = tf.placeholder(tf.int32, [batch_size])

        word_lookup = tf.nn.embedding_lookup(self._words, self._word_in)

        # char_lookup = tf.nn.embedding_lookup(self._chars, self._char_in)

        fw_cell = rnn.LSTMCell(32, initializer=tf.contrib.layers.xavier_initializer())
        bw_cell = rnn.LSTMCell(32, initializer=tf.contrib.layers.xavier_initializer())

        _, rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_lookup, sequence_length=self._w_lens,
                                                     dtype=tf.float16, swap_memory=True)

        hidden = tf.concat([rnn_out[0][1], rnn_out[1][1]], axis=1)
        w = tf.get_variable("w", initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[hidden.shape[1], label_size], dtype=tf.float16)
        b = tf.get_variable("bias", shape=[1], dtype=tf.float16)

        logits = tf.matmul(hidden, w) + b

        if phase == Phase.Validation or phase == Phase.Train:
            self._y = tf.placeholder(tf.int8, shape=[batch_size, label_size], name='lbl')
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self._y, dtype=tf.float16), logits=logits)
            # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(tf.cast(self._y, tf.float32), axis=1),
            # logits=logits)
            self._loss = loss = tf.reduce_sum(losses)

        if phase == Phase.Train:
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

        if phase == Phase.Validation:
            res = tf.argmax(logits, axis=1)

            # get highest scoring classes for all outputs, returns booleans
            correct = tf.equal(res, tf.argmax(self._y, axis=1))

            # get average of the booleans == accuracy
            self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens_w(self):
        return self._w_lens

    # @property
    # def lens_c(self):
    #     return self._c_lens

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def char_in(self):
        return self._char_in

    @property
    def word_in(self):
        return self._word_in

    @property
    def y(self):
        return self._y


class SparseModel:
    def __init__(self,
                 batch_size,
                 w_input_size,
                 c_input_size,
                 n_labels,
                 n_chars,
                 n_words,
                 phase):

        self._char_in = tf.sparse_placeholder(dtype=tf.int32,
                                              shape=np.array([batch_size, c_input_size], dtype=np.int64))
        self._word_in = tf.sparse_placeholder(dtype=tf.int32,
                                              shape=np.array([batch_size, w_input_size], dtype=np.int64))

        self._words = tf.get_variable('words', initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[n_words, 300], dtype=tf.float32)

        self._w_lens = tf.placeholder(tf.int32, [batch_size])
        self._c_lens = tf.placeholder(tf.int32, [batch_size])

        word_lookup = tf.nn.embedding_lookup_sparse(self._words, self._word_in,
                                                    None)  # char_lookup = tf.nn.embedding_lookup(self._chars, self._char_in)

        fw_cell = rnn.LSTMCell(32, initializer=tf.contrib.layers.xavier_initializer())
        bw_cell = rnn.LSTMCell(32, initializer=tf.contrib.layers.xavier_initializer())
        print(self._w_lens)
        word_lookup = tf.reshape(word_lookup, [batch_size, -1, 300])

        print(word_lookup)

        _, rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_lookup, dtype=tf.float32, swap_memory=True,
                                                     sequence_length=self._w_lens)
        print(rnn_out)
        hidden = tf.concat([rnn_out[0][1], rnn_out[1][1]], axis=1)

        print(hidden)

        w = tf.get_variable("w", initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[hidden.shape[1], n_labels], dtype=tf.float32)
        b = tf.get_variable("bias", shape=[1], dtype=tf.float32)

        logits = tf.matmul(hidden, w) + b

        if phase == Phase.Validation or phase == Phase.Train:
            self._y = tf.placeholder(tf.int32, shape=[batch_size], name='lbl')
            labels = tf.reshape(tf.one_hot(self._y, n_labels), [batch_size, -1])

            print(self._y)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, dtype=tf.float32), logits=logits)
            # tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(tf.cast(self._y, tf.float32), axis=1),
            # logits=logits)
            self._loss = loss = tf.reduce_sum(losses)
        if phase == Phase.Train:
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

        if phase == Phase.Validation:
            res = tf.argmax(logits, axis=1)

            # get highest scoring classes for all outputs, returns booleans
            correct = tf.equal(res, tf.argmax(labels, axis=1))
            print(correct)
            # get average of the booleans == accuracy
            self._accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def lens_w(self):
        return self._w_lens

    @property
    def lens_c(self):
        return self._c_lens

    @property
    def values(self):
        return self._values

    @property
    def loss(self):
        return self._loss

    @property
    def probs(self):
        return self._probs

    @property
    def train_op(self):
        return self._train_op

    @property
    def char_in(self):
        return self._char_in

    @property
    def word_in(self):
        return self._word_in

    @property
    def y(self):
        return self._y
