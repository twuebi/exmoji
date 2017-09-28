from enum import IntEnum
import inspect
import pickle
import sys
from os import path
from operator import itemgetter

import nltk
import numpy as np
from lxml import etree

MODEL_DIR = path.dirname(inspect.getfile(inspect.currentframe()))
sys.path.append(path.join(MODEL_DIR, "../../dependencies/"))
from ClassifierBasedGermanTagger.ClassifierBasedGermanTagger import ClassifierBasedGermanTagger


class IOB_Type(IntEnum):
    O = 0
    I = 1
    B = 2


def check_index(index, starts, ends):
    if index in starts:
        for end, categories in starts[index].items():
            if end in ends:
                ends[end] += [[(index, end, category[0]), category] for category in categories]
            else:
                ends[end] = [[(index, end, category[0]), category] for category in categories]

    elif index in ends:
        del ends[index]

    return ends


class Datalist:

    def __init__(self, trained_datalist=None):
        """
        Creates a new Datalist object.
        if no trained datalist is given new numbereres are trained while loading data.

        :param trained_datalist: If not None the given datalist is used as a source instead of creating numberers from scratch
        """
        self.train = trained_datalist is None
        if self.train:
            self.word_nums = Numberer()
            self.emo_nums = Numberer()
            self.max_len_sentences = -1
            self.max_amount_sentences = -1
        else:
            self.word_nums = trained_datalist.word_nums
            self.emo_nums = trained_datalist.emo_nums
            self.max_len_sentences = trained_datalist.max_len_sentences
            self.max_amount_sentences = trained_datalist.max_amount_sentences


class SentimentDatalist(Datalist):

    def __init__(self, trained_datalist=None):
        super().__init__(trained_datalist)

        self.data = []
        if self.train:
            self.char_nums = Numberer()
            self.max_len_char = -1
            self.max_len_word = -1
        else:
            self.char_nums = trained_datalist.char_nums
            self.max_len_char = trained_datalist.max_len_char
            self.max_len_word = trained_datalist.max_len_word

    def load_document_sentiments(self, path):
        with open(path) as f:
            for line in f:
                line = line.replace('\n', '')

                if line:
                    parts = line.split('\t')

                    sents = nltk.sent_tokenize(parts[1], language="german")
                    if len(sents) > self.max_len_sentences:
                        self.max_len_sentences = len(sents)

                    chars = [[self.char_nums.number(c, self.train) for c in sent] for sent in sents]

                    max_sentence_chars = max(map(len, chars))
                    if max_sentence_chars > self.max_len_char:
                        self.max_len_char = max_sentence_chars

                    sentences = [
                        [self.word_nums.number(word, self.train) for word in nltk.word_tokenize(sent, language="german")]
                        for sent in sents
                    ]

                    max_sentence_len = max(map(len, sentences))
                    if max_sentence_len > self.max_len_word:
                        self.max_len_word = max_sentence_len

                    emotion = self.emo_nums.number(parts[3], True)
                    self.data.append((chars, sentences, emotion))

    def create_sentiment_batches(self, batch_size, mode='indices'):
        if mode not in {'indices', 'multi_hot'}:
            raise ValueError("Batch mode must be one of ('indices', 'multi_hot')")

        if mode == 'indices':
            char_width = self.max_len_char
            word_width = self.max_len_word
        else:
            char_width = self.n_chars
            word_width = self.n_words

        char_batch = np.zeros((batch_size, self.max_len_sentences, char_width), dtype=np.uint8)
        word_batch = np.zeros((batch_size, self.max_len_sentences, word_width), dtype=np.uint32)
        sentence_lengths = np.empty(batch_size, dtype=np.uint32)
        chars_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
        words_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
        label_batch = np.empty(batch_size, dtype=np.uint8)

        matrices = [[], []]
        seq_length_batches = [[], [], []]
        labels = []

        for y, (chars, words, emotion) in enumerate(self):
            if y and not y % batch_size:
                #append previous batch data
                matrices[0].append(char_batch)
                matrices[1].append(word_batch)
                seq_length_batches[0].append(chars_seq_length)
                seq_length_batches[1].append(words_seq_length)
                seq_length_batches[2].append(sentence_lengths)
                labels.append(label_batch)

                #create empty representations for the next batch
                char_batch = np.zeros((batch_size, self.max_len_sentences, char_width), dtype=np.uint8)
                word_batch = np.zeros((batch_size, self.max_len_sentences, word_width), dtype=np.uint32)
                sentence_lengths = np.empty(batch_size, dtype=np.uint32)
                chars_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
                words_seq_length = np.zeros((batch_size, self.max_len_sentences), dtype=np.uint32)
                label_batch = np.empty(batch_size, dtype=np.uint8)

            chars_lengths = list(map(len, chars))
            words_lengths = list(map(len, words))
            sentence_length = len(chars)
            
            index = y % batch_size
            for i, (char, word, char_len, word_len) in enumerate(zip(chars, words, chars_lengths, words_lengths)):
                if mode == 'indices':
                    char_batch[index, i, :char_len] = char
                    word_batch[index, i, :word_len] = word
                else:
                    char_batch[index, i, char] = 1
                    word_batch[index, i, word] = 1

            sentence_lengths[index] = sentence_length
            chars_seq_length[index, :sentence_length] = chars_lengths
            words_seq_length[index, :sentence_length] = words_lengths
            label_batch[index] = emotion

        return matrices, seq_length_batches, labels

    @property
    def numberers(self):
        return self.char_nums, self.word_nums, self.emo_nums

    @property
    def n_chars(self):
        return self.char_nums.max

    @property
    def n_words(self):
        return self.word_nums.max

    @property
    def n_labels(self):
        return self.emo_nums.max

    def __iter__(self):
        for entry in self.data:
            yield entry

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "<SentimentDatalist with {} rows>".format(len(self.data))


class AspectDatalistBase(Datalist):

    RELATIVE_POS_PATH = '../../dependencies/nltk_german_classifier_data.pickle'

    def __init__(self, trained_datalist=None):
        super().__init__(trained_datalist)
        
        #get directory part of the path to this module
        with open(path.join(MODEL_DIR, self.RELATIVE_POS_PATH), "rb") as model_file:
            self.pos_tagger = pickle.load(model_file)

        if self.train:
            self.category_nums = Numberer()
            self.polarity_aspect_category_nums = Numberer()
            self.distance_nums = Numberer(first_element=0)
            self.pos_tag_nums = Numberer()
        else:
            self.category_nums = trained_datalist.category_nums
            self.polarity_aspect_category_nums = trained_datalist.polarity_aspect_category_nums
            self.distance_nums = trained_datalist.distance_nums
            self.pos_tag_nums = trained_datalist.pos_tag_nums

    def process_document_text(self, document):
        sentences = [
            #normalize `` and '' to the original " to allow for accurate sequence tagging
            [
                word.replace('``', '"').replace("''", '"') if word in ("''", "``") else word
                for word in nltk.word_tokenize(sentence, language="german")
            ]
            for sentence in nltk.sent_tokenize(document, language="german")
        ]
        pos_tags = self.pos_tagger.tag_sents(sentences)

        single_lengths = [len(sentence) for sentence in sentences]
        sentence_lengths = sum(single_lengths)

        numbered_sentences = []
        numbered_pos_tags = []

        for sentence, sentence_pos in zip(sentences, pos_tags):
            numbered_sentences += [
                self.word_nums.number(word.lower(), self.train) for word in sentence
            ]
            numbered_pos_tags += [
                self.pos_tag_nums.number(pos[1], self.train) for pos in sentence_pos
            ]

        return sentences, numbered_sentences, numbered_pos_tags, sentence_lengths, single_lengths

    def create_iob_batches(self, iob_data, batch_size, mini_batch_size, mini_batch=True, bucketing=True, predict=False):
        if bucketing:
            iob_data = sorted(iob_data, key=itemgetter(-1),reverse=True)
        text_batches = []
        pos_batches = []
        if not predict:
            iob_batches = []
        document_length_batches = []

        num_batches = len(iob_data) // batch_size
        for start, end in zip(
            range(0, (num_batches * batch_size) - batch_size + 1, batch_size),
            range(batch_size, (num_batches * batch_size) + 1, batch_size)
        ):
            text_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)
            pos_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)

            if not predict:
                iob_batch = np.zeros((batch_size, self.max_len_sentences, self.category_nums.max), dtype=np.int32)

            document_lengths = np.zeros(batch_size, dtype=np.int32)
            sentence_lengths = np.zeros([batch_size,self.max_amount_sentences])

            for document_index, (document, iob_markup, pos_tags,single_length,document_length) in enumerate(iob_data[start:end]):
                sentence_lengths[document_index][:len(single_length)] = single_length
                document_lengths[document_index] = min(document_length, self.max_len_sentences)

                if document_length <= self.max_len_sentences:
                    text_batch[document_index, :document_length] = document
                    pos_batch[document_index, :document_length] = pos_tags
                else:
                    text_batch[document_index] = document[:self.max_len_sentences]
                    pos_batch[document_index] = pos_tags[:self.max_len_sentences]

                if not predict:
                    for i, iob in enumerate(iob_markup[:min(document_length, self.max_len_sentences)]):
                        if iob: #if there are aspect annotations on this token
                            iob_batch[document_index, i, iob] = 1
            if mini_batch:
                if predict:
                    iob_batch = None
                iob_batch, text_batch, pos_batch, document_lengths = self.create_mini_batch(batch_size, iob_batch,text_batch,pos_batch,sentence_lengths,mini_batch_size)


            if not predict:
                iob_batches.append(iob_batch)
            text_batches.append(text_batch)
            pos_batches.append(pos_batch)
            document_length_batches.append(document_lengths)

        return (text_batches, iob_batches, document_length_batches, pos_batches) if not predict else (text_batches, document_length_batches, pos_batches)


    def create_mini_batch(self,batch_size ,iob, text, pos, sentence_lengths,max_length):
        prediction = iob is None

        sentence_ratios = (sentence_lengths / max_length).squeeze()
        ceiled_cum_sum = np.insert(np.ceil(np.cumsum(sentence_ratios, axis=1)).astype(np.int32),0,0,axis=1)

        n_splits = np.amax(ceiled_cum_sum)
        if not prediction:
            iob_new = np.zeros([n_splits, batch_size, max_length, self.category_nums.max], dtype=np.int32)
        text_new = np.zeros([n_splits, batch_size, max_length], dtype=np.int64)
        pos_new = np.zeros([n_splits, batch_size, max_length], dtype=np.int32)
        length_new = np.zeros([n_splits, batch_size], dtype=np.int32)

        for x,(single_breakpoints, single_length, single_text, single_iob, single_pos) \
                in enumerate(zip(ceiled_cum_sum, sentence_lengths , text, iob, pos)):

            lengths = np.zeros(shape=[max(n_splits+1,sentence_ratios.shape[1])])
            lengths[:len(single_length)] = single_length

            unique_breaks = np.unique(single_breakpoints)
            breaks = np.zeros(shape=[n_splits+1],dtype=np.int32)

            for num in range(1,len(unique_breaks)):

                breaks[unique_breaks[num-1]:unique_breaks[num]] = np.arange(unique_breaks[num-1],unique_breaks[num])
                breaks[unique_breaks[num]:] = unique_breaks[num]

                breaks[-1] = n_splits

            for i in range(0, n_splits):
                start = breaks[i]
                end = breaks[i+1]

                copy_until = int(np.sum(lengths[start:end]))

                diff = copy_until - max_length
                if diff > 0:
                    copy_until = max_length
                    lengths[start] -= diff
                    lengths[end] += diff

                text_new[i,x,:copy_until] = single_text[i*copy_until:(i+1)*copy_until]
                if not prediction:
                    iob_new[i,x,:copy_until] = single_iob[i*copy_until:(i+1)*copy_until]
                pos_new[i,x,:copy_until] = single_pos[i*copy_until:(i+1)*copy_until]
                length_new[i,x] = copy_until

        return iob_new, text_new, pos_new, length_new

    @property
    def numberers(self):
        return self.category_nums, self.distance_nums, self.word_nums, self.emo_nums, self.pos_tag_nums

    @property
    def n_pos_tags(self):
        return self.pos_tag_nums.max

    @property
    def n_words(self):
        return self.word_nums.max

    @property
    def n_polarities(self):
        return self.emo_nums.max

    @property
    def n_distances(self):
        return self.distance_nums.max

    @property
    def n_categories(self):
        return self.category_nums.max


class AspectDatalist(AspectDatalistBase):

    def __init__(self, trained_datalist=None):
        super().__init__(trained_datalist)

        self.iob_data = []
        self.polarity_data = []

    def load_iob(self, path, verbose=False):
        parser = etree.parse(path)

        for processed_count, element in enumerate(parser.xpath("//Document"), 1):
            element_text = element.xpath("text")[0].text

            sentences, numbered_sentences, numbered_pos_tags, sentence_lengths, single_lengths = self.process_document_text(element_text)
            if self.train and sentence_lengths > self.max_len_sentences:
                self.max_len_sentences = sentence_lengths
            if len(single_lengths) > self.max_amount_sentences:
                self.max_amount_sentences = len(single_lengths)
            annotation_indices = {}
            annotation_to_index = {}
            aspect_polarities = []
            aspect_categories = []

            for opinion in element.xpath(".//Opinion"):
                target = opinion.xpath("@target")[0]
                category = opinion.xpath("@category")[0]
                polarity = opinion.xpath("@polarity")[0]
                category = category[:category.find("#")]

                if target == "NULL":
                    iob_annotation = [[]] * sentence_lengths
                    aspect_locations = np.zeros((1, sentence_lengths))
                    aspect_polarities.append(self.emo_nums.number(polarity, self.train))
                    break
                else:
                    start = int(opinion.xpath("@from")[0])
                    end = int(opinion.xpath("@to")[0])
                    annotation_category = [category, True]
                    aspect_polarities.append(self.emo_nums.number(polarity, self.train))
                    annotation = (start, end, category)

                    if start in annotation_indices:
                        if end in annotation_indices[start] and not annotation in annotation_to_index:
                            annotation_indices[start][end].append(annotation_category)
                        else:
                            annotation_indices[start][end] = [annotation_category]
                    else:
                        annotation_indices[start] = {end : [annotation_category]}

                    if not annotation in annotation_to_index:
                        annotation_to_index[annotation] = len(annotation_to_index)
                        aspect_categories.append(self.polarity_aspect_category_nums.number(category, self.train))

            else:
                #ignore irrelevant documents
                if not annotation_indices:
                    continue

                iob_annotation = []
                #initialize all aspect maps
                aspect_locations = np.ones((len(annotation_to_index), sentence_lengths))

                text_index = 0
                word_index = 0
                available_ends = {}
                new = True

                for sentence in sentences:
                    #advance to next sentence
                    while sentence[0][0] != element_text[text_index]:
                        available_ends = check_index(text_index, annotation_indices, available_ends)
                        text_index += 1

                    for word in sentence:
                        #advance to next word
                        while word[0] != element_text[text_index]:
                            available_ends = check_index(text_index, annotation_indices, available_ends)
                            text_index += 1

                        available_ends = check_index(text_index, annotation_indices, available_ends)
                        iob_annotation.append([])

                        if available_ends:
                            for categories in available_ends.values():
                                for category in categories:
                                    #mark aspect locations
                                    aspect_locations[annotation_to_index[category[0]], word_index] = 0

                                    if category[1][1]: #if the annotation doesn't have a begin element yet
                                        category[1][1] = False
                                        iob_annotation[-1].append(self.category_nums.number((IOB_Type.B, category[1][0]), self.train))
                                    else:
                                        iob_annotation[-1].append(self.category_nums.number((IOB_Type.I, category[1][0]), self.train))

                        text_index += len(word) - 1
                        word_index += 1

            self.iob_data.append((numbered_sentences, iob_annotation, numbered_pos_tags, single_lengths, sentence_lengths))

            for aspect, polarity, category in zip(aspect_locations, aspect_polarities, aspect_categories):
                if np.any(aspect == 0):
                    #get the first and last indices of the array
                    start, end = np.where(aspect == 0)[0][[0, -1]]
                    #add distances from the aspect
                    if start:
                        aspect[:start] = np.fromiter(
                            (self.distance_nums.number(d, self.train) for d in np.arange(-start, 0)),
                            np.int16, start
                        )
                    if end < (len(aspect) - 1):
                        aspect[end + 1:] = np.fromiter(
                            (self.distance_nums.number(d, self.train) for d in np.arange(1, len(aspect) - end)),
                            np.int16, len(aspect) - end - 1
                        )

                self.polarity_data.append((numbered_sentences, aspect, polarity, category, numbered_pos_tags))

            if verbose and processed_count % 100 == 0:
                print("Processed", processed_count, "documents", end="\r")

        if verbose:
            print("Processed", processed_count, "documents")



    def create_aspect_polarity_batches(self, batch_size):
        text_batches = []
        pos_batches = []
        aspect_location_batches = []
        polarity_batches = []
        document_length_batches = []
        aspect_category_batches = []

        num_batches = len(self.polarity_data) // batch_size
        for start, end in zip(
            range(0, (num_batches * batch_size) - batch_size + 1, batch_size),
            range(batch_size, (num_batches * batch_size) + 1, batch_size)
        ):
            text_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)
            pos_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)
            polarity_batch = np.zeros((batch_size, self.emo_nums.max), dtype=np.int32)
            aspect_location_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)
            document_lengths = np.zeros(batch_size, dtype=np.int32)
            aspect_category_batch = np.zeros(batch_size, dtype=np.int32)

            for document_index, (document, aspect_markup, polarity, aspect_category, pos_tags) in enumerate(self.polarity_data[start:end]):
                document_length = len(document)
                document_lengths[document_index] = min(document_length, self.max_len_sentences)
                polarity_batch[document_index, polarity] = 1
                aspect_category_batch[document_index] = aspect_category

                if document_length <= self.max_len_sentences:
                    text_batch[document_index, :document_length] = document
                    pos_batch[document_index, :document_length] = pos_tags
                    aspect_location_batch[document_index, :document_length] = aspect_markup
                else:
                    text_batch[document_index] = document[:self.max_len_sentences]
                    pos_batch[document_index] = pos_tags[:self.max_len_sentences]
                    aspect_location_batch[document_index] = aspect_markup[:self.max_len_sentences]

            text_batches.append(text_batch)
            pos_batches.append(pos_batch)
            polarity_batches.append(polarity_batch)
            aspect_location_batches.append(aspect_location_batch)
            document_length_batches.append(document_lengths)
            aspect_category_batches.append(aspect_category_batch)

        return text_batches, aspect_location_batches, polarity_batches, document_length_batches, aspect_category_batches, pos_batches

    def __repr__(self):
        return "<AspectDatalist with {} iob rows and {} polarity rows>".format(len(self.iob_data), len(self.polarity_data))


class Numberer:

    def __init__(self, first_element=None):
        self.unkown_idx = 1
        if first_element:
            self.num2value = {2 : first_element}
            self.value2num = {first_element : 2}
            self.idx = 3
        else:
            self.num2value = {}
            self.value2num = {}
            self.idx = 2

    def number(self, value, train):

        if train and value not in self.value2num:
            self.value2num[value] = self.idx
            self.num2value[self.idx] = value
            self.idx += 1

        return self.value2num.get(value, self.unkown_idx)

    def __getitem__(self, item):
        return self.value2num[item]

    @property
    def max(self):
        return self.idx

    def value(self, num):
        return self.num2value.get(num, None)
