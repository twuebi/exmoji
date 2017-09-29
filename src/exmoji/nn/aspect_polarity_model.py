import tensorflow as tf

from exmoji.nn import Mode


class AspectPolarityModel():

    def __init__(self, config, maximum_sequence_length, mode):
        self.words = tf.placeholder(tf.int32, shape=[None, config.input_size], name="words")
        self.distances = tf.placeholder(tf.int32, shape=[None, config.input_size], name="distances") 
        self.categories = tf.placeholder(tf.int32, shape=[None], name="categories")
        self.pos = tf.placeholder(tf.int32,shape=[None, config.input_size], name="pos")

        self.fw_initial_state = tf.placeholder(tf.float32, shape=[None, config.hidden_neurons], name="initial_forward")
        self.bw_initial_state = tf.placeholder(tf.float32, shape=[None, config.hidden_neurons], name="initial_backward")

        self.document_lengths = tf.placeholder(tf.int32, shape=[None], name="lengths")

        if mode != mode.PREDICT:
            self.labels = tf.placeholder(tf.float32, shape=[None, config.label_size], name="labels")

        word_embeddings = tf.get_variable("word_embeddings", shape=[config.vocabulary_size, config.word_embedding_size])

        if config.distance_embedding_size:
            distance_embeddings = tf.get_variable("distance_embeddings", shape=[config.num_distances, config.distance_embedding_size])
            embedded_distances = tf.nn.embedding_lookup(distance_embeddings, self.distances)
        if config.pos_embedding_size:
            pos_embeddings = tf.get_variable("pos_embeddings", shape=[config.num_pos, config.pos_embedding_size])
            embedded_pos_tags = tf.nn.embedding_lookup(pos_embeddings, self.pos)
        if config.category_embedding_size:
            category_embeddings = tf.get_variable("category_embeddings", shape=[config.num_categories, config.category_embedding_size])
            embedded_categories = tf.nn.embedding_lookup(category_embeddings, self.categories)

        embedded_words = tf.nn.embedding_lookup(word_embeddings, self.words)

        combined_inputs = embedded_words

        if config.distance_embedding_size:
            # Concatenate word and distance embeddings elementwise
            combined_inputs = tf.concat((embedded_words, embedded_distances), axis=2)
        if config.pos_embedding_size:
            # Dito for POS tags
            combined_inputs = tf.concat((embedded_pos_tags, embedded_distances), axis=2)

        if mode == Mode.TRAIN:
            combined_inputs = tf.nn.dropout(combined_inputs, config.input_dropout)

        hidden = self._bidirectional_rnn(combined_inputs, config, mode, self.fw_initial_state, self.bw_initial_state)

        if config.category_embedding_size:
            hidden = tf.concat((hidden, embedded_categories), axis=1)

        output_weights = tf.get_variable("output_Weight", shape=[hidden.shape[-1], config.label_size])
        output_bias = tf.get_variable("output_bias", shape=[config.label_size])
        # Apply weights on every pair of word representations from the forward and backward propagation
        logits = tf.nn.xw_plus_b(hidden, output_weights, output_bias)

        if mode != Mode.PREDICT:
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
            self.loss = tf.reduce_sum(losses)

        if mode == Mode.TRAIN:
            self.training_operation = tf.train.AdamOptimizer(config.initial_learning_rate).minimize(losses)

        elif mode == Mode.PREDICT:
            self.results = tf.argmax(logits, axis=1, name="results")

        elif mode == Mode.VALIDATE:
            # Highest probability labels of the gold standard data.
            hp_labels = tf.argmax(self.labels, axis=1)

            # Predicted labels
            labels = tf.argmax(logits, axis=1)

            # Calculates labeled accuracy score
            self.equal_counts = tf.equal(hp_labels, labels)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(hp_labels, labels), tf.float32))

    def _bidirectional_rnn(self, input_embeddings, config, mode):
        _, self.state = tf.nn.bidirectional_dynamic_rnn(
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
        # Concatenate forward and backward output cells
        return tf.concat(self.state, axis=1)
