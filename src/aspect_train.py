#!/usr/bin/env python3
from collections import namedtuple
from multiprocessing import Process
import os

import tensorflow as tf
import numpy as np
import argparse

from exmoji.nn import Mode, IOBModel, AspectPolarityModel


IOBConfig = namedtuple("IOBConfig",
    "batch_size label_size input_size word_embedding_size "
    "hidden_neurons hidden_dropout input_dropout initial_learning_rate "
    "max_epochs vocabulary_size pos_embedding_size num_pos mini_batch_size "
    "patience model_path"
)


PolarityConfig = namedtuple("PolarityConfig",
    "batch_size label_size input_size num_distances "
    "vocabulary_size word_embedding_size distance_embedding_size max_epochs "
    "hidden_neurons hidden_dropout input_dropout initial_learning_rate "
    "num_categories category_embedding_size pos_embedding_size num_pos "
    "patience model_path mini_batch_size"
)


def train_iob_model(training_batches, validation_batches, training_max_length, validation_max_length, config):
    with tf.Session() as session:
        with tf.variable_scope("model", reuse=False):
            train_model = IOBModel(config, training_max_length, Mode.TRAIN)

        with tf.variable_scope("model", reuse=True):
            validation_model = IOBModel(config, validation_max_length, Mode.VALIDATE)

        with tf.variable_scope("model", reuse=True):
            prediction_model = IOBModel(config, validation_max_length, Mode.PREDICT)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_loss = -1
        best_accuracy = -1
        patience_accumulator = 0

        for epoch in range(config.max_epochs):
            train_loss = 0
            validation_loss = 0
            validation_accuracy = 0

            for mini_text_batch, mini_iob_batch, mini_length_batch, mini_pos_batch in zip(*training_batches):
                fw_init_state = np.zeros([config.batch_size,config.hidden_neurons])
                bw_init_state = np.zeros([config.batch_size,config.hidden_neurons])

                mini_batch_loss = 0
                for text, iob, length, pos in zip(mini_text_batch, mini_iob_batch, mini_length_batch, mini_pos_batch):
                    ratio = np.sum( np.count_nonzero(iob,axis=0), axis=0) / (np.sum(length))
                    ratio[np.argwhere(ratio == 0)] += 1
                    ratio = np.ones(shape=41)/ratio
                    (fw_init_state,bw_init_state),loss, _ = session.run([train_model.state,train_model.loss, train_model.training_operation],
                        {
                            train_model.ratio : ratio,
                            train_model.fw_initial_state : fw_init_state,
                            train_model.bw_initial_state: bw_init_state,
                            train_model.inputs : text,
                            train_model.labels : iob,
                            train_model.document_lengths : length,
                            train_model.pos : pos
                        }
                    )
                    mini_batch_loss += loss
                mini_batch_loss /= len(mini_text_batch)
                train_loss += mini_batch_loss
            predicted_positives = np.zeros([41])
            true_negatives = np.zeros([41])
            true_positives = np.zeros([41])
            false_negatives = np.zeros([41])
            for mini_text_batch, mini_iob_batch, mini_length_batch, mini_pos_batch in zip(*validation_batches):
                mini_batch_accuracy = 0
                mini_batch_loss = 0
                fw_init_state = np.zeros([config.batch_size, config.hidden_neurons])
                bw_init_state = np.zeros([config.batch_size, config.hidden_neurons])
                for text, iob, length, pos in zip(mini_text_batch, mini_iob_batch, mini_length_batch, mini_pos_batch):
                    ratio = np.ones([config.label_size])
                    (fw_init_state, bw_init_state),loss, accuracy, pred, predictions_vs_labels,\
                    true_pos,\
                    false_pos,\
                    true_neg,\
                    false_neg = session.run([validation_model.state,validation_model.loss, validation_model.accuracy,
                                             validation_model.labels1,validation_model.label_equality,
                                             validation_model.true_pos, validation_model.false_pos,
                                             validation_model.true_neg, validation_model.false_neg],
                        {
                            validation_model.ratio : ratio,
                            validation_model.fw_initial_state : fw_init_state,
                            validation_model.bw_initial_state: bw_init_state,
                            validation_model.inputs : text,
                            validation_model.labels : iob,
                            validation_model.document_lengths : length,
                            validation_model.pos : pos
                        }
                    )

                    true_positives += np.sum(np.reshape(true_pos,[-1,41]),axis=0)
                    predicted_positives += predictions_vs_labels[1]
                    true_negatives += np.sum(np.reshape(true_neg,[-1,41]),axis=0)
                    false_negatives += np.sum(np.reshape(false_neg,[-1,41]),axis=0)
                    mini_batch_loss += loss
                    mini_batch_accuracy += accuracy

                mini_batch_accuracy /= len(mini_text_batch)
                mini_batch_loss /= len(mini_text_batch)
                validation_loss += mini_batch_loss
                validation_accuracy += mini_batch_accuracy
            train_loss /= len(training_batches[0])
            validation_loss /= len(validation_batches[0])
            validation_accuracy /= len(validation_batches[0])
            print()
            print("precision: ")
            print(true_positives.astype(np.int64) / predicted_positives.astype(np.int64))
            print()
            print("recall: ")
            print(true_positives.astype(np.int64) / (false_negatives+true_positives.astype(np.int64)))
            print()
            print(
                "epoch {} | train loss: {:.4f} | validation loss: {:.4f} | Accuracy: {:.2f}%".format(
                    epoch, train_loss, validation_loss, validation_accuracy * 100
                )
            )

            if best_accuracy == -1 or validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                if config.model_path:
                    saver.save(session, os.path.join(config.model_path, os.path.split(config.model_path)[1]))

            if best_loss == -1 or validation_loss < best_loss:
                best_loss = validation_loss

            else:
                patience_accumulator += 1
                if patience_accumulator >= config.patience:
                    print("Early stopping after {} epochs without improvement.".format(patience_accumulator))
                    return


def train_aspect_polarity_model(training_batches, validation_batches, training_max_length, validation_max_length, config):
    with tf.Session() as session:
        with tf.variable_scope("model", reuse=False):
            train_model = AspectPolarityModel(config, training_max_length, Mode.TRAIN)

        with tf.variable_scope("model", reuse=True):
            validation_model = AspectPolarityModel(config, validation_max_length, Mode.VALIDATE)

        with tf.variable_scope("model", reuse=True):
            prediction_model = AspectPolarityModel(config, validation_max_length, Mode.PREDICT)

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_loss = -1
        best_accuracy = -1
        patience_accumulator = 0

        for epoch in range(config.max_epochs):
            train_loss = 0
            validation_loss = 0
            validation_accuracy = 0

            for mini_text_batch, mini_annotation_batch, mini_polarity_batch, mini_length_batch, mini_category_batch, mini_pos_batch in zip(*training_batches):
                fw_init_state = np.zeros([config.batch_size, config.hidden_neurons])
                bw_init_state = np.zeros([config.batch_size, config.hidden_neurons])

                mini_batch_loss = 0
                for text_batch, annotation_batch, polarity_batch, length_batch, category_batch, pos_batch in\
                        zip(mini_text_batch, mini_annotation_batch, mini_polarity_batch, mini_length_batch, mini_category_batch, mini_pos_batch):
                    (fw_init_state, bw_init_state), loss, _ = session.run(
                        [train_model.state, train_model.loss, train_model.training_operation],
                        {
                            train_model.categories: category_batch,
                            train_model.fw_initial_state: fw_init_state,
                            train_model.bw_initial_state: bw_init_state,
                            train_model.words: text_batch,
                            train_model.labels: polarity_batch,
                            train_model.document_lengths: length_batch,
                            train_model.pos: pos_batch,
                            train_model.distances: annotation_batch
                        }
                        )
                    mini_batch_loss += loss

                mini_batch_loss /= len(mini_text_batch)
                train_loss += mini_batch_loss

            for mini_text_batch, mini_annotation_batch, mini_polarity_batch, mini_length_batch, mini_category_batch, mini_pos_batch in zip(*validation_batches):
                fw_init_state = np.zeros([config.batch_size, config.hidden_neurons])
                bw_init_state = np.zeros([config.batch_size, config.hidden_neurons])

                mini_batch_accuracy = 0
                mini_batch_loss = 0
                for text_batch, annotation_batch, polarity_batch, length_batch, category_batch, pos_batch in \
                        zip(mini_text_batch, mini_annotation_batch, mini_polarity_batch, mini_length_batch, mini_category_batch, mini_pos_batch):
                    loss, accuracy, equalities = session.run([validation_model.loss, validation_model.accuracy ,validation_model.equal_counts],
                        {
                            validation_model.categories: category_batch,
                            validation_model.fw_initial_state: fw_init_state,
                            validation_model.bw_initial_state: bw_init_state,
                            validation_model.words: text_batch,
                            validation_model.labels: polarity_batch,
                            validation_model.document_lengths: length_batch,
                            validation_model.pos: pos_batch,
                            validation_model.distances: annotation_batch
                        }
                    )
                    mini_batch_loss += loss
                    mini_batch_accuracy += accuracy

                mini_batch_accuracy /= len(mini_text_batch)
                mini_batch_loss /= len(mini_text_batch)
                validation_loss += mini_batch_loss
                validation_accuracy += mini_batch_accuracy

            train_loss /= len(training_batches[0])
            validation_loss /= len(validation_batches[0])
            validation_accuracy /= len(validation_batches[0])

            print(
                "epoch {} | train loss: {:.4f} | validation loss: {:.4f} | Accuracy: {:.2f}%".format(
                    epoch, train_loss, validation_loss, validation_accuracy * 100
                )
            )

            if best_accuracy == -1 or validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                if config.model_path:
                    saver.save(session, os.path.join(config.model_path, os.path.split(config.model_path)[1]))

            if best_loss == -1 or validation_loss < best_loss:
                best_loss = validation_loss

            else:
                patience_accumulator += 1
                if patience_accumulator >= config.patience:
                    print("Early stopping after {} epochs without improvement.".format(patience_accumulator))
                    return


def print_process(text, dots=4):
    clear_text = " " * (len(text) + dots)
    for num_dots in cycle(range(dots)):
        print(clear_text, end="\r")
        print(text, "." * num_dots, end="\r", sep="")
        sleep(.2)


def load_iob_batches(train_datalist, validation_datalist, batch_size, mini_batch_size):
    loading = Process(target=print_process, args=("Creating batches",))
    loading.start()
    # Makes sure printing process terminates even if something goes wrong
    try:
        training_batches = train_datalist.create_iob_batches(train_datalist.iob_data,batch_size, mini_batch_size)
        validation_batches = validation_datalist.create_iob_batches(validation_datalist.iob_data,batch_size, mini_batch_size)
    except:
        #reraise exception - will still execute finally
        raise
    finally:
        loading.terminate()

    return training_batches, validation_batches


def load_aspect_polarity_batches(train_datalist, validation_datalist, batch_size, mini_batch_size):
    loading = Process(target=print_process, args=("Creating batches",))
    loading.start()
    # Makes sure printing process terminates even if something goes wrong
    try:
        training_batches = train_datalist.create_aspect_polarity_batches(batch_size,mini_batch_size)
        validation_batches = validation_datalist.create_aspect_polarity_batches(batch_size,mini_batch_size)
    except:
        #reraise exception - will still execute finally
        raise
    finally:
        loading.terminate()

    return training_batches, validation_batches


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="a tool for training the exmoji model for extracting and classifying categorical sentiment aspects from documents"
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('train_file', type=argparse.FileType('rb'), help='Processed pickle file for training')
    common_parser.add_argument('validation_file', type=argparse.FileType('rb'), help='Processed pickle file for validation')
    common_parser.add_argument('--batch-size', '-b', metavar='N', type=int, default=512, help='size of training and validation batches')
    common_parser.add_argument('--hidden-neurons', '-n', metavar='N', type=int, default=125, help='number of gru cell neurons')
    common_parser.add_argument('--input-dropout', metavar='N', type=float, default=1, help='dropout retention rate applied to the input')
    common_parser.add_argument('--hidden-dropout', metavar='N', type=float, default=1, help='dropout retention rate applied to bi-rnn gru cells')
    common_parser.add_argument('--learning-rate', '-l', metavar='N', type=float, default=0.001, help='initial learning rate for the Adam optimizer')
    common_parser.add_argument('--max-epochs', '-m', metavar='N', type=int, default=1000, help='maximum epochs before stopping training')
    common_parser.add_argument('--pos-embedding-size', '-p', metavar='N', type=int, default=0, help='size of part of speech (POS) embedding vectors - 0 to disable')
    common_parser.add_argument('--save-model', '-s', metavar='PATH', type=str, default=None, help="path to a model directory the best model is saved to which is created if it doesn't already exist. Defaults to not saving")
    common_parser.add_argument('--early-stopping-patience', '-e', metavar='N', type=str, default=1,
        help='number of epochs to wait for improvements after validation loss increases before stopping and saving the best model')

    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True

    iob_parser = subparsers.add_parser(
        'iob', help="trains sentiment aspect annotation", parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    iob_parser.add_argument('--word-embedding-size', '-w', metavar='N', type=int, default=200, help='size of input word embedding vectors')
    iob_parser.add_argument('--mini-batch-size', '-mb', metavar='N', type=int, default=150, help='size of minibatches')

    polarity_parser = subparsers.add_parser(
        'polarity', help="trains polarity classification of aspects", parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    polarity_parser.add_argument('--word-embedding-size', '-w', metavar='N', type=int, default=200, help='size of input word embedding vectors')
    polarity_parser.add_argument('--distance-embedding-size', '-d', metavar='N', type=int, default=10, help='size of distance embedding vectors - 0 to disable')
    polarity_parser.add_argument('--category-embedding-size', '-c', metavar='N', type=int, default=10, help='size of category embedding vectors - 0 to disable')
    polarity_parser.add_argument('--mini-batch-size', '-mb', metavar='N', type=int, default=150, help='size of minibatches')

    return parser.parse_args()


if __name__ == '__main__':
    from time import sleep
    from itertools import cycle
    import pickle

    arguments = parse_arguments()
    
    with arguments.train_file as in_file:
        train_datalist = pickle.load(in_file)

    with arguments.validation_file as in_file:
        validation_datalist = pickle.load(in_file)

    #Create model directory if it doesn't exist already
    if arguments.save_model:
        if not os.path.exists(arguments.save_model):
            os.makedirs(arguments.save_model)

    if arguments.model == "iob":
        config = IOBConfig(
            batch_size=arguments.batch_size,
            label_size=train_datalist.n_categories,
            mini_batch_size=arguments.mini_batch_size,
            input_size=arguments.mini_batch_size if arguments.mini_batch_size else train_datalist.max_len_sentences,
            word_embedding_size=arguments.word_embedding_size,
            hidden_neurons=arguments.hidden_neurons,
            input_dropout=arguments.input_dropout,
            hidden_dropout=arguments.hidden_dropout,
            initial_learning_rate=arguments.learning_rate,
            max_epochs=arguments.max_epochs,
            vocabulary_size=train_datalist.n_words,
            pos_embedding_size=arguments.pos_embedding_size,
            num_pos=train_datalist.pos_tag_nums.max,
            patience=arguments.early_stopping_patience,
            model_path=arguments.save_model
        )

        training_batches, validation_batches = load_iob_batches(train_datalist, validation_datalist,config.batch_size, config.mini_batch_size)
        # ESC[2K clears the line
        print("\x1b[2KTraining started")
        train_iob_model(
            training_batches, validation_batches, train_datalist.max_len_sentences, validation_datalist.max_len_sentences, config
        )

    else:
        config = PolarityConfig(
            batch_size=arguments.batch_size,
            mini_batch_size=arguments.mini_batch_size,
            label_size=train_datalist.n_polarities,
            input_size=arguments.mini_batch_size if arguments.mini_batch_size else train_datalist.max_len_sentences,
            word_embedding_size=arguments.word_embedding_size,
            distance_embedding_size=arguments.distance_embedding_size,
            hidden_neurons=arguments.hidden_neurons,
            input_dropout=arguments.input_dropout,
            hidden_dropout=arguments.hidden_dropout,
            initial_learning_rate=arguments.learning_rate,
            max_epochs=arguments.max_epochs,
            vocabulary_size=train_datalist.n_words,
            num_distances=train_datalist.n_distances,
            num_categories=train_datalist.polarity_aspect_category_nums.max,
            category_embedding_size=arguments.category_embedding_size,
            pos_embedding_size=arguments.pos_embedding_size,
            num_pos=train_datalist.pos_tag_nums.max,
            patience=arguments.early_stopping_patience,
            model_path=arguments.save_model
        )

        training_batches, validation_batches = load_aspect_polarity_batches(train_datalist, validation_datalist, config.batch_size, config.mini_batch_size)
        # ESC[2K clears the line
        print("\x1b[2KTraining started")
        train_aspect_polarity_model(
            training_batches, validation_batches, train_datalist.max_len_sentences, validation_datalist.max_len_sentences, config
        )
