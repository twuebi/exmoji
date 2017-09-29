#!/bin/python3
import argparse
import pickle

from exmoji.processing.loader import AspectDatalist
import gensim

parser = argparse.ArgumentParser(description="preprocesses and saves data for use in aspect_train")
parser.add_argument('train_file', type=argparse.FileType('r'), help='xml file for training')
parser.add_argument('validation_file', type=argparse.FileType('r'), help='xml file for validation')
parser.add_argument('train_output_file', type=argparse.FileType('wb'), help='output file name of the processed training data')
parser.add_argument('validation_output_file', type=argparse.FileType('wb'), help='output file name of the processed validation data')

arguments = parser.parse_args()

print("Loading Data")
#model = gensim.models.KeyedVectors.load_word2vec_format('german.model', binary=True)
word2index = None #{word: i for i, word in enumerate(model.index2word)}
train_datalist = AspectDatalist(word2id=word2index)

train_datalist.load_iob(arguments.train_file, verbose=True)

with arguments.train_output_file as out_file:
    pickle.dump(train_datalist, out_file)

validation_datalist = AspectDatalist(train_datalist,word2id=word2index)
validation_datalist.load_iob(arguments.validation_file ,verbose=True)
with arguments.validation_output_file as out_file:
    pickle.dump(validation_datalist, out_file)
