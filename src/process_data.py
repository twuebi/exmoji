#!/bin/python2
import argparse
import pickle

from exmoji import AspectDatalist, AspectDatalistBase


parser = argparse.ArgumentParser(description="preprocesses and saves data for use in aspect_train")
parser.add_argument('train_file', type=argparse.FileType('r'), help='xml file for training')
parser.add_argument('validation_file', type=argparse.FileType('r'), help='xml file for validation')
parser.add_argument('train_output_file', type=argparse.FileType('wb'), help='output file name of the processed training data')
parser.add_argument('validation_output_file', type=argparse.FileType('wb'), help='output file name of the processed validation data')
parser.add_argument('prediction_output_file', type=argparse.FileType('wb'), help='output file name of a prediction file for processing data in predict')

arguments = parser.parse_args()

print("Loading Data")
train_datalist = AspectDatalist()
train_datalist.load_iob(arguments.train_file, verbose=True)
with arguments.train_output_file as out_file:
    pickle.dump(train_datalist, out_file)

validation_datalist = AspectDatalist(train_datalist)
validation_datalist.load_iob(arguments.validation_file, verbose=True)
with arguments.validation_output_file as out_file:
    pickle.dump(validation_datalist, out_file)

with arguments.prediction_output_file as out_file:
    pickle.dump(AspectDatalistBase(train_datalist), out_file)
