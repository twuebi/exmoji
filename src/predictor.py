#!/usr/bin/env python3
import json
from collections import namedtuple
from os import path

import tensorflow as tf
import numpy as np

from exmoji import IOB_Type


class ModelWrapper():
 
    def __init__(self, model_name):
        self._graph = tf.Graph()
        self._model_name = model_name
        with self._graph.as_default():
            self._loader = tf.train.import_meta_graph("{}.meta".format(path.join(model_name, path.split(model_name)[1])))
 
    def __enter__(self):
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._loader.restore(self._session, tf.train.latest_checkpoint(self._model_name))
            self._get_variables()
 
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()


IOBVars = namedtuple("IOBVars", "rnn_output_states inputs lengths pos initial_forward initial_backward results")
PolarityVars = namedtuple("PolarityVars", "words lengths pos distances categories results")


class IOBModelWrapper(ModelWrapper):

    def __init__(self, model_name, datalist):
        super().__init__(model_name)
        self._datalist = datalist
        self._iobs = np.array([(IOB_Type.O, "NULL")] + [item[1] for item in sorted(datalist.category_nums.num2value.items(), key=lambda x: x[0])])

    def _get_variables(self):
        self._vars = IOBVars(
            inputs=self._graph.get_tensor_by_name("model_2/inputs:0"),
            lengths=self._graph.get_tensor_by_name("model_2/lengths:0"),
            pos=self._graph.get_tensor_by_name("model_2/pos:0"),
            initial_forward=self._graph.get_tensor_by_name("model_2/initial_forward:0"),
            initial_backward=self._graph.get_tensor_by_name("model_2/initial_backward:0"),
            results=self._graph.get_tensor_by_name("model_2/results:0"),
            rnn_output_states=self._graph.get_tensor_by_name("model_2/bi_rnn:0")
        )
        self._hidden_neurons = self._vars.initial_forward.shape[1]

    def classify_batch(self, mini_text_batch, mini_length_batch, mini_pos_batch, batch_size):
        fw_init_state = np.zeros([batch_size, self._hidden_neurons])
        bw_init_state = np.zeros([batch_size, self._hidden_neurons])

        mini_batch_loss = 0
        for text, length, pos in zip(mini_text_batch, mini_length_batch, mini_pos_batch):
            (fw_init_state, bw_init_state), results = self._session.run([self._vars.rnn_output_states, self._vars.results],
                {
                    self._vars.initial_forward : fw_init_state,
                    self._vars.initial_backward : bw_init_state,
                    self._vars.inputs : text,
                    self._vars.lengths : length,
                    self._vars.pos : pos
                }
            )
            for word_nums, iob in zip(text, results):
                aspect_locations = np.unique(np.nonzero(results)[0])
                words = word_nums[:word_nums.nonzero()[0][-1]]
                
                if not aspect_locations.size:
                    yield [IOB_Type.O] * words.size
                else:
                    #TODO: improve this (including improved I and B handling and distances for polarity)
                    aspects = []
                    open_aspects = {}
                    for i, (word_iob, word) in enumerate(zip(iob, words)):
                        if np.count_nonzero(word_iob):
                            for current in self._iobs[word_iob.nonzero()]:
                                markup_type, category = current
                                if category in open_aspects and markup_type == IOB_Type.I:
                                    aspects[open_aspects[category]].append(category)

                                open_aspects[category] = len(aspects)
                                aspects.append(([None] * i) + [category])


                    yield aspects


class PolarityModelWrapper(ModelWrapper):

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_variables(self):
        self._vars = PolarityVars(
            words=self._graph.get_tensor_by_name("model_2/words:0"),
            lengths=self._graph.get_tensor_by_name("model_2/lengths:0"),
            pos=self._graph.get_tensor_by_name("model_2/pos:0"),
            distances=self._graph.get_tensor_by_name("model_2/distances:0"),
            categories=self._graph.get_tensor_by_name("model_2/categories:0"),
            results=self._graph.get_tensor_by_name("model_2/results:0")
        )


def classify_iob(model, documents, datalist, batch_size, mini_batch_size):
    vectors = [datalist.process_document_text(document)[1:] for document in documents]

    yield from model.classify_batch(*datalist.create_iob_batches(vectors, batch_size, mini_batch_size, mini_batch=False, predict=True), batch_size)


if __name__ == '__main__':
    import sys
    import pickle

    import argparse

    from exmoji import AspectDatalistBase


    parser = argparse.ArgumentParser(description="Annotates sentiment aspects and classifies their polarities")
    parser.add_argument("iob_model_name", help="name of a trained iob model file")
    parser.add_argument("polarity_model_name", help="name of a trained polarity model file")
    parser.add_argument("prediction_datalist", type=argparse.FileType("rb"), help="path to a prediction datalist")
    parser.add_argument("input", nargs="?", default=sys.stdin, type=argparse.FileType("r"),
        help="path to an input file of documents seperated by newlines; defaults to stdin")
    parser.add_argument("--output", "-o", metavar="PATH", default=sys.stdout, type=argparse.FileType("w"),
        help="path to an output file prediction results are saved to")
    parser.add_argument("--batch-size", "-b", metavar="N", default=1, type=int,
        help="maximum batch size. builds up a buffer of documents until batch size is reached"
            "or EOF is found. Handles a final batch with a size smaller than batch size if necessary.")
    parser.add_argument("--mini-batch-size", "-mb", metavar="N", default=150, type=int,
        help="size of mini batches")
    arguments = parser.parse_args()

    with arguments.prediction_datalist as datalist_file:
        datalist = pickle.load(datalist_file)

    line_buffer = []
    with IOBModelWrapper(arguments.iob_model_name, datalist) as iob_model, arguments.input as input_file, arguments.output as output_file:
        for line in input_file:
            line_buffer.append(line)
            if len(line_buffer) == arguments.batch_size:
                output_file.write(
                    "\n".join(map(str, classify_iob(iob_model, line_buffer, datalist, len(line_buffer), arguments.mini_batch_size)))
                    + "\n"
                )
                line_buffer = []

        # Handles batch overflow
        if line_buffer:
            output_file.write(
                "\n".join(map(str, classify_iob(iob_model, line_buffer, datalist, len(line_buffer), arguments.mini_batch_size)))
                + "\n"
            )
