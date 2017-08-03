import numpy as np
import tensorflow as tf

from exmoji.nn.model import SentenceModel, Model, Phase, SparseModel
from sklearn import metrics


def train(training_set, validation_set):
    n_chars = training_set.n_chars
    n_words = training_set.n_words
    n_labels = training_set.n_labels
    max_number_of_sentences = training_set.max_len_sentences
    max_length_sentence = training_set.max_len_word

    batch_size = 32
    n_words_per_sent = 200
    n_sents_per_doc = 100

    train_matrices, train_lengths, train_labels = training_set.create_batches(batch_size, 'indices')
    val_matrices, val_lengths, val_labels = validation_set.create_batches(batch_size, 'indices')

    train_chars, train_words = train_matrices
    train_char_seq_lengths, train_words_seq_lengths, train_sentence_lengths = train_lengths


    val_chars, val_words = val_matrices
    val_char_seq_lengths, val_words_seq_lengths, val_sentence_lengths = val_lengths

    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=False):
            train_model = SentenceModel(
                batch_size=batch_size,
                sentences_size=max_number_of_sentences,
                w_input_size=max_length_sentence,
                label_size=n_labels,
                n_words_per_sent=n_words_per_sent,
                n_sents_per_doc=n_sents_per_doc,
                n_words=n_words,
                phase=Phase.Train
            )

        with tf.variable_scope('model', reuse=True):
            val_model = SentenceModel(
                batch_size=batch_size,
                sentences_size=max_number_of_sentences,
                w_input_size=max_length_sentence,
                label_size=n_labels,
                n_words_per_sent=n_words_per_sent,
                n_sents_per_doc=n_sents_per_doc,
                n_words=n_words,
                phase=Phase.Validation
            )

        sess.run(tf.global_variables_initializer())

        for epoch in range(128):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0

            pre = []
            true = []
            # Train on all batches.
            for batch in range(len(train_words)):
                # loss, _ = sess.run([train_model.loss, train_model.train_op], {
                #     train_model.word_in: train_w[batch], train_model.lens_w: train_w_len[batch],train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch], train_model.y: train_labels[batch]})
                pred, labels, loss, _ = sess.run(
                    [train_model.predictions, train_model.labels, train_model.loss, train_model.train_op], {
                        # train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch],
                        train_model.sent_in: train_words[batch][:, :n_sents_per_doc, :n_words_per_sent],
                        train_model.lens_w: train_words_seq_lengths[batch],
                        train_model.lens_w: train_words_seq_lengths[batch],
                        train_model.y: train_labels[batch]})
                train_loss += loss
                print("train", batch)

            # validation on all batches.
            for batch in range(len(val_words)):
                # loss, _ = sess.run([val_model.loss, val_model.val_op], {
                #     val_model.word_in: val_w[batch], val_model.lens_w: val_w_len[batch],val_model.char_in: val_c[batch], val_model.lens_c: val_c_len[batch], val_model.y: val_labels[batch]})
                losses, labels, loss, acc, pred = sess.run(
                    [val_model.losses, val_model.labels, val_model.loss, val_model.accuracy, val_model.predictions], {
                        val_model.sent_in: val_words[batch][:, :n_sents_per_doc, :n_words_per_sent],
                        val_model.lens_w: val_words_seq_lengths[batch],
                        val_model.y: val_labels[batch]})
                validation_loss += loss
                accuracy += acc
                print(val_labels[batch])
                print(pred)
                true.extend(val_labels[batch])
                pre.extend(pred)
            print(metrics.confusion_matrix(true, pre, labels=[0, 1, 2]))
            val_conf = metrics.precision_recall_fscore_support(true, pre, labels=[0, 1, 2], average='macro')
            train_loss /= len(train_words)
            validation_loss /= len(val_words)
            accuracy /= len(val_words)
            print(val_conf)
            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100))


def train_sparse(training_set, validation_set):
    n_chars = max(training_set.n_chars, validation_set.n_chars)
    n_words = max(training_set.n_words, validation_set.n_words)
    n_labels = max(training_set.n_labels, validation_set.n_labels)
    max_w = max(training_set.max_len_word, validation_set.max_len_word)
    max_c = max(training_set.max_len_char, validation_set.max_len_char)
    batch_size = 128
    print("coo start")
    train_matrices, train_seq_lengths, train_labels = training_set.create_coo_batches(batch_size)
    val_matrices, val_seq_lengths, val_labels = validation_set.create_coo_batches(batch_size)

    char_train_mat, word_train_mat = train_matrices
    char_val_mat, word_val_mat = val_matrices

    val_c_len, val_w_len = val_seq_lengths
    train_c_len, train_w_len = train_seq_lengths

    # lens = max(len(train_w), len(val_w))
    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=False):
            train_model = SparseModel(
                batch_size=batch_size,
                c_input_size=max_c,
                w_input_size=max_w,
                n_labels=n_labels,
                n_chars=n_chars,
                n_words=n_words,
                phase=Phase.Train)
        with tf.variable_scope('model', reuse=True):
            val_model = SparseModel(
                batch_size=batch_size,
                c_input_size=max_c,
                w_input_size=max_w,
                n_labels=n_labels,
                n_chars=n_chars,
                n_words=n_words,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(2048):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            # Train on all batches.
            for batch in range(len(word_train_mat)):
                # loss, _ = sess.run([train_model.loss, train_model.train_op], {
                #     train_model.word_in: train_w[batch], train_model.lens_w: train_w_len[batch],train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch], train_model.y: train_labels[batch]})
                x_in = tf.SparseTensorValue(
                    np.array([word_train_mat[batch].row, word_train_mat[batch].col]).T,
                    word_train_mat[batch].data,
                    word_train_mat[batch].shape
                )
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    # train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch],
                    train_model.word_in: x_in, train_model.lens_w: train_w_len[batch],
                    train_model.y: train_labels[batch]})

                train_loss += loss

            # validation on all batches.
            for batch in range(len(word_val_mat)):
                # loss, _ = sess.run([val_model.loss, val_model.val_op], {
                #     val_model.word_in: val_w[batch], val_model.lens_w: val_w_len[batch],val_model.char_in: val_c[batch], val_model.lens_c: val_c_len[batch], val_model.y: val_labels[batch]})
                loss, acc = sess.run([val_model.loss, val_model.accuracy], {
                    val_model.word_in: tf.SparseTensorValue(
                        np.array([word_val_mat[batch].row, word_val_mat[batch].col]).T,
                        word_val_mat[batch].data,
                        word_val_mat[batch].shape
                    ), val_model.lens_w: val_w_len[batch],
                    val_model.y: val_labels[batch]})
                validation_loss += loss
                accuracy += acc
            train_loss /= len(word_train_mat)
            validation_loss /= len(word_val_mat)
            accuracy /= len(word_val_mat)

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100))
