from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

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

        self._char_in = tf.placeholder(dtype=tf.int32,name='char_in',shape=[batch_size, c_input_size])
        self._word_in = tf.placeholder(name='words_in', dtype=tf.int32, shape=[batch_size, w_input_size])

        self._words = tf.get_variable('words', initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[n_words, 1], dtype=tf.float16)

        self._chars = tf.get_variable('chars', initializer=tf.contrib.layers.xavier_initializer(),
                                      shape=[n_chars, 20], dtype=tf.float32)

        self._w_lens = tf.placeholder(tf.int32,[batch_size])
        self._c_lens = tf.placeholder(tf.int32, [batch_size])

        word_lookup = tf.nn.embedding_lookup(self._words, self._word_in)
        char_lookup = tf.nn.embedding_lookup(self._chars, self._char_in)

        fw_cell = rnn.LSTMCell(32,initializer=tf.contrib.layers.xavier_initializer())
        bw_cell = rnn.LSTMCell(32, initializer=tf.contrib.layers.xavier_initializer())

        _ , rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,char_lookup,sequence_length=self._c_lens,dtype=tf.float32,swap_memory=True)

        hidden = tf.concat([rnn_out[0][1],rnn_out[1][1]], axis=1)
        w = tf.get_variable("w", initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[hidden.shape[1], label_size])
        b = tf.get_variable("bias", shape=[1])

        logits = tf.matmul(hidden,w) + b

        if phase == Phase.Validation or phase == Phase.Train:
            self._y = tf.placeholder(tf.int8, shape=[batch_size, label_size], name='lbl')
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self._y,dtype=tf.float32),logits=logits)
            #tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(tf.cast(self._y, tf.float32), axis=1),
                                                                #logits=logits)
            self._loss = loss = tf.reduce_sum(losses)
        if phase == Phase.Train:

            self._train_op = tf.train.AdamOptimizer().minimize(loss)

        if phase == Phase.Validation:
            res = tf.argmax(logits, axis=1)

            # get highest scoring classes for all outputs, returns booleans
            correct = tf.equal(res, tf.argmax(self._y, axis=1))

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
        return  self._word_in

    @property
    def y(self):
        return self._y
