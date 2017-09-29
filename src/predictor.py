#!/bin/python3
import json

import tensorflow as tf


class ModelWrapper():
 
    def __init__(self, model_name):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._loader = tf.train.import_meta_graph("{}.meta".format(model_name))
 
    def __enter__(self):
        self._session = tf.Session(graph=self._graph)
        with self._graph.as_default():
            self._loader.restore(self._session, tf.train.latest_checkpoint("./"))
            self._get_variables()
 
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()


class IOBModelWrapper(ModelWrapper):

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_variables(self):
        pass


class PolarityModelWrapper(ModelWrapper):

    def __init__(self, model_name):
        super().__init__(model_name)

    def _get_variables(self):
        pass


def classify_iob(documents, datalist, batch_size, mini_batch_size):
    vectors = [datalist.process_document_text(document)[1:] for document in documents]

    return datalist.create_iob_batches(vectors, batch_size, mini_batch_size, mini_batch=False, predict=True)


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
    with arguments.input as input_file, arguments.output as output_file:
        for line in input_file:
            line_buffer.append(line)
            if len(line_buffer) == arguments.batch_size:
                output_file.write("{}\n".format(classify_iob(line_buffer, datalist, arguments.batch_size, arguments.mini_batch_size)))
                line_buffer = []

        # Handles batch overflow
        if line_buffer:
            output_file.write("{}\n".format(classify_iob(line_buffer, datalist, len(line_buffer), arguments.mini_batch_size)))
