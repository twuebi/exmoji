from collections import namedtuple
from multiprocessing import Process

import tensorflow as tf

from exmoji.nn.iob_model import Mode, IOBModel
from exmoji.processing.loader import Datalist


Config = namedtuple("Config",
    "batch_size label_size input_size embedding_size "
    "hidden_neurons hidden_dropout input_dropout initial_learning_rate "
    "max_epochs vocabulary_size"
)


def train(training_batches, validation_batches, training_max_length, validation_max_length, config):
    with tf.Session() as session:
        with tf.variable_scope("model", reuse=False):
            train_model = IOBModel(config, training_max_length, Mode.TRAIN)

        with tf.variable_scope("model", reuse=True):
            validation_model = IOBModel(config, validation_max_length, Mode.VALIDATE)

        session.run(tf.global_variables_initializer())

        for epoch in range(config.max_epochs):
            train_loss = 0
            validation_loss = 0
            validation_accuracy = 0

            for text_batch, iob_batch, length_batch in zip(*training_batches):
                loss, _ = session.run([train_model.loss, train_model.training_operation],
                    {
                        train_model.inputs : text_batch,
                        train_model.labels : iob_batch,
                        train_model.document_lengths : length_batch
                    }
                )
                train_loss += loss

            for text_batch, iob_batch, length_batch in zip(*validation_batches):
                loss, accuracy = session.run([validation_model.loss, validation_model.accuracy],
                    {
                        validation_model.inputs : text_batch,
                        validation_model.labels : iob_batch,
                        validation_model.document_lengths : length_batch
                    }
                )
                validation_loss += loss
                validation_accuracy += accuracy


            train_loss /= len(training_batches[0])
            validation_loss /= len(training_batches[0])

            print(
                "epoch {} | train loss: {:.4f} | validation loss: {:.4f} | Accuracy: {:.2f}%".format(
                    epoch, train_loss, validation_loss, validation_accuracy * 100
                )
            )


if __name__ == '__main__':
    from time import sleep
    from itertools import cycle


    TRAINING_FILE = "../data/train_v1.4.xml"
    VALIDATION_FILE = "../data/dev_v1.4.xml"

    print("Loading Data")
    train_datalist = Datalist()
    train_datalist.load_iob(TRAINING_FILE, verbose=True)
    validation_datalist = Datalist(train_datalist)
    validation_datalist.load_iob(VALIDATION_FILE, verbose=True)

    config = Config(
        batch_size=256,
        label_size=train_datalist.category_nums.max(),
        input_size=train_datalist.max_len_sentences,
        embedding_size=200,
        hidden_neurons=120,
        input_dropout=.95,
        hidden_dropout=.8,
        initial_learning_rate=.001,
        max_epochs=100,
        vocabulary_size=train_datalist.word_nums.max()
    )

    def print_process(text, dots=4):
        clear_text = " " * (len(text) + dots)
        for num_dots in cycle(range(dots)):
            print(clear_text, end="\r")
            print(text, "." * num_dots, end="\r", sep="")
            sleep(.2)

    loading = Process(target=print_process, args=("Creating batches",))
    loading.start()
    # Makes sure printing process terminates even if something goes wrong
    try:
        training_batches = train_datalist.create_iob_batches(config.batch_size)
        validation_batches = validation_datalist.create_iob_batches(config.batch_size)
    finally:
        loading.terminate()

    # ESC[2K clears the line
    print("\x1b[2KTraining started")
    train(training_batches, validation_batches, train_datalist.max_len_sentences, validation_datalist.max_len_sentences, config)
