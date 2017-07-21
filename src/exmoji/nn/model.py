from enum import Enum

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class Phase(Enum):
    Train = 0
    Validation = 1
    Predict = 2

class Model:
    def __init__(self,
                 batch_size,
                 w_input_size,
                 c_input_size,
                 shapes,
                 c_length,
                 w_length,
                 labels,
                 n_chars,
                 n_words,
                 phase):
        label_size = labels.shape[2]
        #self._char_in = tf.placeholder(dtype=tf.int32,name='char_in',shape=[batch_size, c_input_size])
        self._word_in = tf.placeholder(name='words_in', dtype=tf.int64, shape=None)
        w_in = tf.reshape(self._word_in,[-1,2])
        self._values = tf.placeholder(name='values',dtype=tf.int64, shape=None)
        v_in = tf.reshape(self._values,[-1])
        print(w_in)
        print(v_in)
        self._word1 = tf.SparseTensor(w_in,v_in, np.array([batch_size, w_input_size],dtype=np.int64))
        print(self._word1)
        self._words = tf.get_variable('words', initializer=tf.contrib.layers.xavier_initializer(),
                                       shape=[n_words, 100], dtype=tf.float32)
        print(self._words)
        # self._chars = tf.get_variable('chars', initializer=tf.contrib.layers.xavier_initializer(),
        #                               shape=[n_chars, 20], dtype=tf.float32)

        self._w_lens = tf.placeholder(tf.int32,[batch_size])
        #self._c_lens = tf.placeholder(tf.int32, [batch_size])

        word_lookup = tf.nn.embedding_lookup_sparse(self._words, self._word1,None,combiner='sqrtn')        #char_lookup = tf.nn.embedding_lookup(self._chars, self._char_in)

        fw_cell = rnn.LSTMCell(64,initializer=tf.contrib.layers.xavier_initializer())
        bw_cell = rnn.LSTMCell(64, initializer=tf.contrib.layers.xavier_initializer())
        #
        print(word_lookup)
        word_lookup = tf.reshape(word_lookup,[batch_size,-1,100])
        print(word_lookup)
        # filte = tf.get_variable('W_conv', [4, 100, 100])
        # b_c = tf.get_variable('b_conv', [100])
        # rnn_input = tf.nn.conv1d(word_lookup, filte, 1, padding='VALID')
        # rnn_input = tf.nn.relu(rnn_input + b_c)
        # print(rnn_input)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 2, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # print(rnn_input)
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)
        #
        # filte1 = tf.get_variable('W_conv1', [2, 100, 50])
        # b_c1 = tf.get_variable('b_conv1', [50])
        # rnn_input = tf.nn.conv1d(rnn_input, filte1, 1, padding='VALID')
        # rnn_input = tf.nn.relu(rnn_input + b_c1)
        # print(rnn_input)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 2, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # print(rnn_input)
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)
        #
        # filte2 = tf.get_variable('W_conv2', [2, 50, 25])
        # b_c2 = tf.get_variable('b_conv2', [25])
        # rnn_input = tf.nn.conv1d(rnn_input, filte2, 1, padding='VALID')
        # rnn_input = tf.nn.relu(rnn_input + b_c2)
        # print(rnn_input)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 2, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # print(rnn_input)
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)
        #
        # filte3 = tf.get_variable('W_conv3', [2, 25, 12])
        # b_c3 = tf.get_variable('b_conv3', [12])
        # rnn_input = tf.nn.conv1d(rnn_input, filte3, 1, padding='VALID')
        # rnn_input = tf.nn.relu(rnn_input + b_c3)
        # print(rnn_input)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 2, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # print(rnn_input)
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)

        _ , rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,word_lookup,dtype=tf.float32,swap_memory=True)

        # filte = tf.get_variable('W_conv', [50, 20, 25])
        # b_c = tf.get_variable('b_conv', [50])
        # print(char_lookup)
        # rnn_input = tf.nn.conv1d(char_lookup, filte, 20, padding='VALID')
        # rnn_input = tf.nn.relu(rnn_input + b_c)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 1, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)
        #
        # filte = tf.get_variable('W_conv1', [20, 50, 25])
        # b_c = tf.get_variable('b_conv1', [25])
        # print(rnn_input)
        # rnn_input = tf.nn.conv1d(rnn_input, filte, 10, padding='VALID')
        # print(rnn_input)
        # rnn_input = tf.nn.relu(rnn_input + b_c)
        # rnn_input = tf.nn.max_pool(
        #     tf.expand_dims(rnn_input, axis=0),
        #     strides=[1, 1, 1, 1],
        #     ksize=[1, 1, 2, 1],
        #     padding='VALID')
        # rnn_input = tf.squeeze(rnn_input, axis=0)
        #
        # print(rnn_input)
        # if phase == Phase.Train:
        #     rnn_input = tf.nn.dropout(rnn_input, 0.9)

        _, rnn_out = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, word_lookup, dtype=tf.float32, swap_memory=True)

        hidden = tf.concat([rnn_out[0][1],rnn_out[1][1]], axis=1)
        w = tf.get_variable("w", initializer=tf.contrib.layers.xavier_initializer(),
                            shape=[hidden.shape[1], label_size],dtype=tf.float32)
        b = tf.get_variable("bias", shape=[1],dtype=tf.float32)

        logits = tf.matmul(hidden,w) + b

        if phase == Phase.Validation or phase == Phase.Train:
            self._y = tf.placeholder(tf.int32, shape=[batch_size, label_size], name='lbl')
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
        return  self._word_in

    @property
    def y(self):
        return self._y
