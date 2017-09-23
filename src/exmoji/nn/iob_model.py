import tensorflow as tf

from exmoji.nn import Mode


class IOBModel():
    def __init__(self, config, maximum_sequence_length, mode):
        self.inputs = tf.placeholder(tf.int32, shape=[config.batch_size, config.input_size], name="inputs")

        self.document_lengths = tf.placeholder(tf.int32, shape=[config.batch_size], name="lengths")

        if mode != mode.PREDICT:
            self.labels = tf.placeholder(tf.float32,
                                         shape=[config.batch_size, maximum_sequence_length, config.label_size],
                                         name="labels")

        embeddings = tf.get_variable("embeddings", shape=[config.vocabulary_size, config.embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
        input_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)

        if mode == Mode.TRAIN:
            input_embeddings = tf.nn.dropout(input_embeddings, config.input_dropout)

        hidden = self._bidirectional_rnn(input_embeddings, config, mode)

        output_weights = tf.get_variable("output_Weight", shape=[hidden.shape[-1], config.label_size],
                                         initializer=tf.contrib.layers.xavier_initializer())

        self.rnn_out = output_bias = tf.get_variable("output_bias", shape=[config.label_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())

        # Apply weights on every pair of word representations from the forward and backward propagation
        logits = tf.einsum('ijk,kl->ijl', hidden, output_weights)
        nz = tf.count_nonzero(logits, 2)
        greater = tf.greater(nz, 0)
        greater = tf.expand_dims(greater, -1)
        logits = output_bias * tf.cast(greater, tf.float32) + logits

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

            # Calculates labeled accuracy score#
            label_equality = tf.cast(tf.equal(hp_labels, labels), tf.float32)
            label_equality = tf.reduce_sum(label_equality, axis=1)
            diff = maximum_sequence_length - tf.cast(self.document_lengths, dtype=tf.float32)
            quant = label_equality - diff
            denom = (maximum_sequence_length - diff)
            self.accuracy = tf.reduce_mean(quant / denom)

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
