#!/usr/bin/env python3
import argparse
import pickle

from exmoji import AspectDatalist, AspectDatalistBase
from gensim.models.wrappers import FastText
import numpy as np

parser = argparse.ArgumentParser(description="preprocesses and saves data for use in aspect_train")
parser.add_argument('train_file', type=argparse.FileType('r'), help='xml file for training')
parser.add_argument('validation_file', type=argparse.FileType('r'), help='xml file for validation')
parser.add_argument('train_output_file', type=argparse.FileType('wb'), help='output file name of the processed training data')
parser.add_argument('validation_output_file', type=argparse.FileType('wb'), help='output file name of the processed validation data')
parser.add_argument('prediction_output_file', type=argparse.FileType('wb'), help='output file name of a prediction file for processing data in predict')
parser.add_argument('--fasttext_embeddings',metavar='N', type=str, default=None, help='path to file containing pretrained fasttext embeddings'
                                                                   'intersects them with the training vocabulary'
                                                                   'use this option of your memory is limited'
                                                                  'fasttext embeddings are available here:'
                                                                  'https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md')
parser.add_argument('--output_intersected_matrix', metavar='N',type=argparse.FileType('wb'),default=None, help='file to save the intersected fasttext embeddings to')
arguments = parser.parse_args()

print("Loading Data")
train_datalist = AspectDatalist()
train_datalist.load_iob(arguments.train_file, verbose=True)

if arguments.fasttext_embeddings and arguments.output_intersected_matrix:
    out_matrix = np.zeros((train_datalist.n_words,300))
    val2num = train_datalist.word_nums.value2num

    fasttext = FastText.load_fasttext_format(arguments.fasttext_embeddings)

    for key in val2num:
        try:
            out_matrix[val2num[key]] = fasttext[key]
        except KeyError:
            print(key,"not in vocabulary")
    with arguments.output_intersected_matrix as f:
        pickle.dump(out_matrix, f)

with arguments.train_output_file as out_file:
    pickle.dump(train_datalist, out_file)

validation_datalist = AspectDatalist(train_datalist)
validation_datalist.load_iob(arguments.validation_file, verbose=True)
with arguments.validation_output_file as out_file:
    pickle.dump(validation_datalist, out_file)

with arguments.prediction_output_file as out_file:
    pickle.dump(AspectDatalistBase(train_datalist), out_file)
