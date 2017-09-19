from enum import IntEnum

import tensorflow as tf


class Mode(IntEnum):
    TRAIN = 0
    VALIDATE = 1
    PREDICT = 2


class IOBModel():

    def __init__(self, config, maximum_sequence_length, mode):
        self.inputs = tf.placeholder(tf.int32, shape=[config.batch_size, config.input_size], name="inputs")

        self.document_lengths = tf.placeholder(tf.int32, shape=[config.batch_size], name="lengths")

        if mode != mode.PREDICT:
            self.labels = tf.placeholder(tf.float32, shape=[config.batch_size, maximum_sequence_length, config.label_size], name="labels")

        embeddings = tf.get_variable("embeddings", shape=[config.vocabulary_size, config.embedding_size])
        input_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)

        if mode == Mode.TRAIN:
            input_embeddings = tf.nn.dropout(input_embeddings, config.input_dropout)

        hidden = self._bidirectional_rnn(input_embeddings, config, mode)

        output_weights = tf.get_variable("output_Weight", shape=[hidden.shape[-1], config.label_size])
        output_bias = tf.get_variable("output_bias", shape=[config.label_size])
        # Apply weights on every pair of word representations from the forward and backward propagation
        logits = tf.einsum('ijk,kl->ijl', hidden, output_weights) + output_bias

        if mode != Mode.PREDICT:
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.loss = tf.reduce_sum(losses)

        if mode == Mode.TRAIN:
            self.training_operation = tf.train.AdamOptimizer(config.initial_learning_rate).minimize(losses)

        elif mode == Mode.VALIDATE:
            # Highest probability labels of the gold standard data.
            hp_labels = tf.argmax(self.labels, axis=2)

            # Predicted labels
            labels = tf.argmax(logits, axis=2)

            # Calculates labeled accuracy score
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(hp_labels, labels), tf.float32))

    def _rnn(self, input_embeddings, config, mode):
        outputs, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(config.hidden_neurons),
                output_keep_prob=config.hidden_dropout if mode == Mode.TRAIN else 1
            ),
            input_embeddings, sequence_length=self.document_lengths, dtype=tf.float32
        )

        return outputs

    def _bidirectional_rnn(self, input_embeddings, config, mode):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(config.hidden_neurons),
                output_keep_prob=config.hidden_dropout if mode == Mode.TRAIN else 1
            ),
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(config.hidden_neurons),
                output_keep_prob=config.hidden_dropout if mode == Mode.TRAIN else 1
            ),
            input_embeddings, sequence_length=self.document_lengths, dtype=tf.float32
        )
        # Concatenate forward and backward propagated cells pairwise
        return tf.concat(outputs, axis=2)
