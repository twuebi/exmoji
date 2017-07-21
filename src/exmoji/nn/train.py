import numpy as np
import tensorflow as tf

from exmoji.nn.model import Model, Phase

def train(training_set, validation_set):
    n_chars = max(training_set.n_chars, validation_set.n_chars)
    n_words = max(training_set.n_words, validation_set.n_words)
    n_labels = max(training_set.n_labels, validation_set.n_labels)
    max_w = max(training_set.max_len_word, validation_set.max_len_word)
    max_c = max(training_set.max_len_char, validation_set.max_len_char)
    batch_size = 128
    print("coo start")
    train_matrices, train_seq_lengths, train_labels = training_set.create_coo_batches(batch_size)
    val_matrices, val_seq_lengths, val_labels = validation_set.create_coo_batches(batch_size)

    char_train_mat ,word_train_mat = train_matrices
    char_val_mat, word_val_mat = val_matrices

    val_c_len,val_w_len = val_seq_lengths
    train_c_len,train_w_len = train_seq_lengths

    # lens = max(len(train_w), len(val_w))
    print(len(word_train_mat))
    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=False):
            train_model = Model(
                batch_size=batch_size,
                c_input_size=max_c,
                w_input_size=max_w,
                n_labels=n_labels,
                n_chars=n_chars,
                n_words=n_words,
                phase=Phase.Train)
        with tf.variable_scope('model', reuse=True):
            val_model = Model(
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
                loss, accuracy = sess.run([val_model.loss, val_model.accuracy], {
                    val_model.word_in: tf.SparseTensorValue(
                        np.array([word_val_mat[batch].row, word_val_mat[batch].col]).T,
                        word_val_mat[batch].data,
                        word_val_mat[batch].shape
                    ), val_model.lens_w: val_w_len[batch],
                    val_model.y: val_labels[batch]})
                validation_loss += loss
            train_loss /= len(word_train_mat)
            validation_loss /= len(word_val_mat)
            accuracy /= len(word_val_mat)

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100))
