import numpy as np
import tensorflow as tf

from exmoji.nn.model import Model, Phase


def generate_instances(
        set,max_w,max_c,
        batch_size=1024):

    n_batches = len(set.data) // batch_size

    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            set.n_labels),
        dtype=np.int8)
    lengths_w = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int64)
    lengths_c = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int64)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size*max_w),
        dtype=np.int32)
    chars = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_c),
        dtype=np.uint16)
    b = []
    for batch in range(n_batches):
        batch_list = []
        sent_pos = []
        value_list = []
        for idx in range(batch_size):
            position_in_batch = idx
            c = set.data[(batch * batch_size) + idx][0]
            for position_in_sent,word in enumerate(set.data[(batch * batch_size) + idx][1]):
                batch_list.append(position_in_batch)
                sent_pos.append(position_in_sent)
                value_list.append(word)
            #words[batch, idx, :len(w)] = w
            chars[batch, idx,:len(c)] = c


            labels[batch, idx,set.data[(batch * batch_size) + idx][2]] = 1

            # Sequence
            timesteps_w = min(set.max_len_word, len(set.data[(batch * batch_size) + idx][1]))
            timesteps_c = min(set.max_len_char, len(c))
            # Sequence length (time steps)
            lengths_w[batch, idx] = timesteps_w
            lengths_c[batch, idx] = timesteps_c
            # Word characters
        b.append([[batch_list,sent_pos],[value_list]])
    return (np.array(b), chars ,lengths_w, lengths_c , labels)


def train(training_set, validation_set):
    n_chars = max(training_set.n_chars, validation_set.n_chars)
    n_words = max(training_set.n_words,validation_set.n_words)

    max_w = max(training_set.max_len_word,validation_set.max_len_word)
    max_c = max(training_set.max_len_char, validation_set.max_len_char)
    batch_size = 64



    train_w, train_c, train_w_len,train_c_len, train_labels = generate_instances(training_set, max_w, max_c,batch_size)
    val_w, val_c, val_w_len,val_c_len, val_labels = generate_instances(validation_set, max_w, max_c,batch_size)

    lens = max(len(train_w), len(val_w))

    with tf.Session() as sess:
        with tf.variable_scope('model',reuse=False):
            train_model = Model(
                 batch_size=batch_size,
                 shapes=[lens, 2],
                 c_input_size=max_c,
                 w_input_size=max_w,
                 c_length=train_c_len,
                 w_length=train_w_len,
                 labels=train_labels,
                 n_chars=n_chars,
                 n_words=n_words,
                 phase=Phase.Train)
        with tf.variable_scope('model',reuse=True):
            val_model = Model(
                 batch_size=batch_size,
                 c_input_size=max_c,
                 w_input_size=max_w,
                 shapes=[lens, 2],
                 c_length=val_c_len,
                 w_length=val_w_len,
                 labels=val_labels,
                 n_chars=n_chars,
                 n_words=n_words,
                 phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(50):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0

            # Train on all batches.
            for batch in range(train_w.shape[0]):
                # loss, _ = sess.run([train_model.loss, train_model.train_op], {
                #     train_model.word_in: train_w[batch], train_model.lens_w: train_w_len[batch],train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch], train_model.y: train_labels[batch]})
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    #train_model.char_in: train_c[batch], train_model.lens_c: train_c_len[batch],
                    train_model.word_in: train_w[batch][0], train_model.values: train_w[batch][1], train_model.lens_w: train_w_len[batch],
                    train_model.y: train_labels[batch]})

                train_loss += loss

            # validation on all batches.
            for batch in range(val_w.shape[0]):
                # loss, _ = sess.run([val_model.loss, val_model.val_op], {
                #     val_model.word_in: val_w[batch], val_model.lens_w: val_w_len[batch],val_model.char_in: val_c[batch], val_model.lens_c: val_c_len[batch], val_model.y: val_labels[batch]})
                loss, accuracy = sess.run([val_model.loss, val_model.accuracy], {
                    val_model.word_in: val_w[batch][0],val_model.values: val_w[batch][1] ,val_model.lens_w: val_w_len[batch],
                    #val_model.char_in: val_c[batch], val_model.lens_c: val_c_len[batch],
                    val_model.y: val_labels[batch]})
                validation_loss += loss

            train_loss /= train_w.shape[0]
            validation_loss /= val_w.shape[0]
            accuracy /= val_w.shape[0]

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f, validation acc: %.2f" %
                (epoch, train_loss, validation_loss, accuracy * 100))
