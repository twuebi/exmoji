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
PolarityVars = namedtuple("PolarityVars", "words lengths pos distances categories results initial_forward initial_backward rnn_output_states")


class IOBModelWrapper(ModelWrapper):

    def __init__(self, model_name, datalist):
        super().__init__(model_name)
        self._datalist = datalist
        self._categories = np.array(
            [(IOB_Type.O, "NULL")] + [item[1] for item in sorted(datalist.category_nums.num2value.items(), key=lambda x: x[0])]
        )
        self._iobs = np.array([item[1] for item in sorted(datalist.IOB_nums.num2value.items(), key=lambda x: x[0])], dtype=np.object)

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

    def classify_batch(self, text_batch, length_batch, pos_batch, batch_size):
        for mini_text_batch, mini_length_batch, mini_pos_batch in zip(text_batch, length_batch, pos_batch):
            fw_init_state = np.zeros([batch_size, self._hidden_neurons])
            bw_init_state = np.zeros([batch_size, self._hidden_neurons])

            mini_batch_loss = 0
            for text, lengths, pos in zip(mini_text_batch, mini_length_batch, mini_pos_batch):
                (fw_init_state, bw_init_state), results = self._session.run([self._vars.rnn_output_states, self._vars.results],
                    {
                        self._vars.initial_forward : fw_init_state,
                        self._vars.initial_backward : bw_init_state,
                        self._vars.inputs : text,
                        self._vars.lengths : lengths,
                        self._vars.pos : pos
                    }
                )
                for word_nums, iob, length in zip(text, results, lengths):
                    words = word_nums[:length]
                    if not np.count_nonzero(iob):
                        yield None
                    else:
                        aspects = []
                        aspect_categories = []

                        open_aspects = {}
                        for i, (categories, word) in enumerate(zip(iob, words)):
                            zeros = np.where(categories == 0)[0]
                            non_zeros = categories.nonzero()[0]
                            
                            for markup_type, category, index in zip(categories[non_zeros], self._categories[non_zeros], non_zeros):
                                markup_type = self._iobs[markup_type]
                                if markup_type == IOB_Type.B or category not in open_aspects:
                                    open_aspects[category] = len(aspects)
                                    aspects.append(np.ones(mini_text_batch.shape[2]))
                                    aspect_categories.append(self._datalist.category_nums[category])
                                    aspects[-1][i] = 0
                                else:
                                    aspects[open_aspects[category]][i] = 0

                            for category in categories[zeros]:
                                if category in open_aspects:
                                    del category

                        yield text, lengths, pos, aspects, aspect_categories


class PolarityModelWrapper(ModelWrapper):

    def __init__(self, model_name, datalist):
        super().__init__(model_name)
        self._datalist = datalist

    def _get_variables(self):
        self._vars = PolarityVars(
            words=self._graph.get_tensor_by_name("model_2/words:0"),
            lengths=self._graph.get_tensor_by_name("model_2/lengths:0"),
            pos=self._graph.get_tensor_by_name("model_2/pos:0"),
            distances=self._graph.get_tensor_by_name("model_2/distances:0"),
            categories=self._graph.get_tensor_by_name("model_2/categories:0"),
            results=self._graph.get_tensor_by_name("model_2/results:0"),
            initial_forward=self._graph.get_tensor_by_name("model_2/initial_forward:0"),
            initial_backward=self._graph.get_tensor_by_name("model_2/initial_backward:0"),
            rnn_output_states=self._graph.get_tensor_by_name("model_2/bi_rnn:0")
        )
        self._hidden_neurons = self._vars.initial_forward.shape[1]

    def classify_batch(self, mini_text_batch, mini_length_batch, mini_pos_batch, mini_annotation_batch, mini_category_batch, batch_size):
        fw_init_state = np.zeros([batch_size, self._hidden_neurons])
        bw_init_state = np.zeros([batch_size, self._hidden_neurons])

        mini_batch_loss = 0
        for text_batch, annotation_batch, length_batch, category_batch, pos_batch in zip(
            mini_text_batch, mini_annotation_batch, mini_length_batch, mini_category_batch, mini_pos_batch
        ):
            (fw_init_state, bw_init_state), results = self._session.run(
                [self._vars.rnn_output_states, self._vars.results],
                {
                    self._vars.words : np.squeeze(text_batch,0),
                    self._vars.distances : annotation_batch,
                    self._vars.categories : category_batch,
                    self._vars.lengths : np.squeeze(length_batch,0),
                    self._vars.pos : np.squeeze(pos_batch,0),
                    self._vars.initial_forward : fw_init_state,
                    self._vars.initial_backward : bw_init_state
                }
            )

            yield from (self._datalist.emo_nums.value(polarity) for polarity in results)


def classify_iob(model, documents, datalist, batch_size, mini_batch_size):
    vectors = [datalist.process_document_text(document)[1:] for document in documents]

    document_to_aspect_indices = []
    all_aspects = []
    aspect_batch = []

    for aspects in model.classify_batch(*datalist.create_iob_batches(vectors, batch_size, mini_batch_size, predict=True), batch_size):
        if not aspects or not aspects[-1]:
            document_to_aspect_indices.append([])
            continue

        document_indices = []
        aspect_markup = []

        for aspect, category in zip(*aspects[3:]):
            # get the first and last indices of the array
            start, end = np.where(aspect == 0)[0][[0, -1]]
            #Add text markup
            aspect_markup.append(["I" if i == 0 else "O" for i in aspect])
            # add distances from the aspect
            if start:
                aspect[:start] = np.fromiter(
                    (datalist.distance_nums[d] for d in np.arange(-start, 0)),
                    np.int16, start
                )
            if end < (len(aspect) - 1):
                aspect[end + 1:] = np.fromiter(
                    (datalist.distance_nums[d] for d in np.arange(1, len(aspect) - end)),
                    np.int16, len(aspect) - end - 1
                )

            document_indices.append(len(all_aspects))
            all_aspects.append((*aspects[:3], aspect, category))

        document_to_aspect_indices.append(document_indices)

    batch = [[np.array(i)] for i in zip(*all_aspects)]
    return document_to_aspect_indices, aspect_markup, list(polarity_model.classify_batch(*batch, len(all_aspects)))


def aspects_to_string(indices, markup, polarities):
    for index_list in indices:
        if not index_list:
            yield "NONE\tNeutral\n\n"
        else:
            for i in index_list
                yield "{}\t{}\n\n".format(markup[i], polarities[i])

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
    with IOBModelWrapper(arguments.iob_model_name, datalist) as iob_model, \
        PolarityModelWrapper(arguments.polarity_model_name, datalist) as polarity_model, \
        arguments.input as input_file, arguments.output as output_file:
        for line in input_file:
            line_buffer.append(line)
            if len(line_buffer) == arguments.batch_size:
                for entry in aspects_to_string(*classify_iob(iob_model, line_buffer, datalist, len(line_buffer), arguments.mini_batch_size)):
                    output_file.write(entry)
                line_buffer = []

        # Handles batch overflow
        if line_buffer:
            for entry in aspects_to_string(*classify_iob(iob_model, line_buffer, datalist, len(line_buffer), arguments.mini_batch_size)):
                output_file.write(entry)
