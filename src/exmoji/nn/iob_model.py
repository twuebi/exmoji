import tensorflow as tf

from exmoji.nn import Mode
from tensorflow.python.layers.core import Dense

class IOBModel():

    def __init__(self, config, maximum_sequence_length, mode):
        maximum_sequence_length=config.mini_batch_size
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
        self.document_lengths = tf.placeholder(tf.int32, shape=[None], name="lengths")
        self.pos = tf.placeholder(tf.int32,shape=[None, None], name="pos")
        self.fw_initial_state = tf.placeholder(tf.float32,shape=[None,config.hidden_neurons], name="initial_forward")
        self.bw_initial_state = tf.placeholder(tf.float32, shape=[None, config.hidden_neurons], name="initial_backward")

        if mode != mode.PREDICT:
            self.labels = tf.placeholder(tf.float32,
                                         shape=[None, None, config.label_size],
                                         name="labels")

        embeddings = tf.get_variable("embeddings", shape=[config.vocabulary_size, config.word_embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
        input_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs)

        if config.pos_embedding_size:
            pos_embeddings = tf.get_variable("pos_embeddings", shape=[config.num_pos, config.pos_embedding_size],
                                     initializer=tf.contrib.layers.xavier_initializer())

            input_embeddings = tf.concat((input_embeddings, tf.nn.embedding_lookup(pos_embeddings, self.pos)), axis=2)

        if mode == Mode.TRAIN:
            input_embeddings = tf.nn.dropout(input_embeddings, config.input_dropout)

        hidden = self._bidirectional_rnn(input_embeddings, config, mode, self.fw_initial_state, self.bw_initial_state)


        output_weights = tf.get_variable("output_Weight", shape=[hidden.shape[-1], config.label_size],
                                         initializer=tf.contrib.layers.xavier_initializer())

        output_bias = tf.get_variable("output_bias", shape=[config.label_size],
                                                     initializer=tf.contrib.layers.xavier_initializer())



        # Apply weights on every pair of word representations from the forward and backward propagation
        logits = tf.einsum('ijk,kl->ijl', hidden, output_weights)
        nz = tf.count_nonzero(logits, 2)
        greater = tf.greater(nz, 0)
        greater_exp = tf.expand_dims(greater, -1)
        self.logits = output_bias * tf.cast(greater_exp, tf.float32) + logits

        if mode != Mode.PREDICT:
            self.ratio = tf.placeholder(tf.float32,shape=[config.label_size],name="class_weights")
            losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.labels,pos_weight=self.ratio, logits=logits)
            self.loss = tf.reduce_sum(losses)

        if mode == Mode.TRAIN:
            self.training_operation = tf.train.AdamOptimizer(config.initial_learning_rate).minimize(losses)

        elif mode == Mode.PREDICT:
            self.results = tf.round(tf.nn.sigmoid(logits), name="results")

        elif mode == Mode.VALIDATE:
            # Highest probability labels of the gold standard data.
            hp_labels = self.labels

            # Predicted labels
            self.labels1  = labels = tf.round(tf.nn.sigmoid(logits))

            # Calculates labeled accuracy score#
            self.label_equality = tf.reduce_sum(tf.count_nonzero(hp_labels,axis=0),axis=0),tf.reduce_sum(tf.count_nonzero(labels,axis=0),axis=0)

            #self.true_pos = tf.boolean_mask(tf.cast(tf.equal(hp_labels, labels), tf.float32), tf.greater(labels, 0))

            self.true_pos = tf.logical_and(tf.greater(hp_labels,0),tf.greater(labels,0))
            self.false_pos = tf.logical_and(tf.less(hp_labels,1),tf.greater(labels,0))

            self.true_neg = tf.logical_and(tf.less(hp_labels, 1), tf.less(labels, 1))
            self.false_neg = tf.logical_and(tf.greater(hp_labels, 0), tf.less(labels, 1))

            self.false = tf.reduce_sum(tf.add(tf.cast(self.false_pos, tf.float32), tf.cast(self.false_neg, tf.float32)))
            self.total_labels = tf.cast(tf.reduce_sum(self.document_lengths) * config.label_size, tf.float32)
            self.ham = 1 / self.total_labels * self.false

            label_equality = tf.boolean_mask(tf.cast(tf.equal(hp_labels, labels), tf.float32),tf.greater(tf.count_nonzero(greater,axis=1),0))
            masked_lengths = tf.boolean_mask(tf.cast(self.document_lengths, dtype=tf.float32),tf.greater(tf.count_nonzero(greater,axis=1),0))


            label_equality = tf.reduce_sum(label_equality, axis=1)
            diff = maximum_sequence_length - masked_lengths
            quant = label_equality - tf.expand_dims(diff, -1)
            denom = (maximum_sequence_length - diff)
            self.accuracy = tf.reduce_mean(quant / tf.expand_dims(denom, -1))

    def _bidirectional_rnn(self, input_embeddings, config, mode, fw_state, bw_state):
        outputs, self.state = tf.nn.bidirectional_dynamic_rnn(
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(config.hidden_neurons),
                output_keep_prob=config.hidden_dropout if mode == Mode.TRAIN else 1
            ),
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(config.hidden_neurons),
                output_keep_prob=config.hidden_dropout if mode == Mode.TRAIN else 1
            ),
            input_embeddings, sequence_length=self.document_lengths, dtype=tf.float32,
            initial_state_fw=fw_state,
            initial_state_bw=bw_state
        )
        self.state = tf.identity(self.state, name="bi_rnn")
        # Concatenate forward and backward propagated cells pairwise
        return tf.concat(outputs, axis=2)
