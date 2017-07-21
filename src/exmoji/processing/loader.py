import nltk
import numpy as np

class Data:

    def __init__(self,train):
        self.data = []
        self.train = train
        self.char_nums = Numberer()
        self.word_nums = Numberer()

        self.emo_nums = Numberer()

    def load(self,path):
        with open(path) as f:
            for line in f:
                line = line.replace('\n','')
                if line != '':
                    parts = line.split('\t')
                    chars = np.array([self.char_nums.number(c,self.train) for c in parts[1]],dtype=np.int8)
                    words = np.array([self.word_nums.number(word,self.train) for word in nltk.word_tokenize(parts[1])],dtype=np.int32)
                    emotion = self.emo_nums.number(parts[3],self.train)
                    self.data.append((chars,words,emotion))

class Numberer:

    def __init__(self):
        self.num2value = {}
        self.value2num = {}
        self.unkown_idx = 0
        self.idx = 1

    def number(self, value, train):

        if value not in self.value2num and train:
            self.value2num[value] = self.idx
            self.num2value[self.idx] = value
            self.idx += 1

        return self.value2num.get(value,self.unkown_idx)

    def max(self):
        return self.idx-1

    def value(self,num):
        return self.num2value.get(num,None)