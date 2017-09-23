from enum import IntEnum

import nltk
import numpy as np
from lxml import etree


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

    __slots__ = (
        "data", "train", "char_nums", "word_nums", "emo_nums", "category_nums", "distance_nums",
        "max_len_char", "max_len_word", "max_len_sentences"
    )

    def __init__(self, trained_datalist=None):
        """
        Creates a new Datalist object.
        if no pretrained numberers are given new numbereres are trained while loading.

        :param trained_numberers: A tuple of (char_nums, word_nums, emo_nums, category_nums) or None
        """
        self.data = []
        self.train = trained_datalist is None
        if self.train:
            self.char_nums = Numberer()
            self.word_nums = Numberer()
            self.emo_nums = Numberer()
            self.category_nums = Numberer()
            self.distance_nums = Numberer(first_element=0)
            self.max_len_char = -1
            self.max_len_word = -1
            self.max_len_sentences = -1

        else:
            self.char_nums = trained_datalist.char_nums
            self.word_nums = trained_datalist.word_nums
            self.emo_nums = trained_datalist.emo_nums
            self.category_nums = trained_datalist.category_nums
            self.distance_nums = trained_datalist.distance_nums
            self.max_len_char = trained_datalist.max_len_char
            self.max_len_word = trained_datalist.max_len_word
            self.max_len_sentences = trained_datalist.max_len_sentences

    def load_iob(self, path, verbose=False):
        parser = etree.parse(path)
        word_tokenizer = nltk.TweetTokenizer()

        self.data = [[], []]

        for processed_count, element in enumerate(parser.xpath("//Document"), 1):
            element_text = element.xpath("text")[0].text

            sentences = [
                #normalize `` and '' to the original " to allow for accurate sequence tagging
                [
                    word.replace('``', '"').replace("''", '"') if word in ("''", "``") else word
                    for word in nltk.word_tokenize(sentence, language="german")
                ]
                for sentence in nltk.sent_tokenize(element_text, language="german")
            ]

            sentence_lengths = sum(map(len, sentences))
            if self.train and sentence_lengths > self.max_len_sentences:
                self.max_len_sentences = sentence_lengths

            numbered_sentences = []
            for sentence in sentences:
                numbered_sentences += [
                    self.word_nums.number(word, self.train) for word in sentence
                ]

            annotation_indices = {}
            annotation_to_index = {}
            aspect_polarities = []

            for opinion in element.xpath(".//Opinion"):
                target = opinion.xpath("@target")[0]
                category = opinion.xpath("@category")[0]
                polarity = opinion.xpath("@polarity")[0]
                category = category[:category.find("#")]

                if target == "NULL":
                    iob_annotation = np.ones(sentence_lengths, dtype=np.int32) * self.category_nums.number((IOB_Type.I, category), self.train)
                    iob_annotation[0] = self.category_nums.number((IOB_Type.B, category), self.train)
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
                        else:
                            iob_annotation[-1].append(self.category_nums.number(IOB_Type.O, self.train))

                        text_index += len(word) - 1
                        word_index += 1


                # TODO: improve multi annotation handling
                # Only keeps first annotation layer at the moment, discarding overlapping ones
                iob_annotation = np.array([cat[0] for cat in iob_annotation])

            self.data[0].append((numbered_sentences, iob_annotation))
            for aspect, polarity in zip(aspect_locations, aspect_polarities):
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

                self.data[1].append((numbered_sentences, aspect, polarity))

            if verbose and processed_count % 100 == 0:
                print("Processed", processed_count, "documents", end="\r")

        if verbose:
            print("Processed", processed_count, "documents")

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

    def create_iob_batches(self, batch_size):
        
        text_batches = []
        iob_batches = []
        document_length_batches = []

        num_batches = len(self.data[0]) // batch_size
        for start, end in zip(
            range(0, (num_batches * batch_size) - batch_size + 1, batch_size),
            range(batch_size, (num_batches * batch_size) + 1, batch_size)
        ):
            text_batch = np.zeros((batch_size, self.max_len_sentences), dtype=np.int32)
            iob_batch = np.zeros((batch_size, self.max_len_sentences, self.category_nums.max()), dtype=np.int32)
            document_lengths = np.zeros(batch_size, dtype=np.int32)

            for document_index, (document, iob_markup) in enumerate(self.data[0][start:end]):
                document_length = len(document)
                document_lengths[document_index] = max(document_length, self.max_len_sentences)
                if document_length <= self.max_len_sentences:
                    text_batch[document_index, :document_length] = document
                else:
                    text_batch[document_index] = document[:self.max_len_sentences]

                for i, iob in enumerate(iob_markup[:max(document_length, self.max_len_sentences)]):
                    iob_batch[document_index, i, iob] = 1

            iob_batches.append(iob_batch)
            text_batches.append(text_batch)
            document_length_batches.append(document_lengths)

        return text_batches, iob_batches, document_length_batches

    def create_aspect_polarity_batches(self, batch_size):
        text_batches = []
        aspect_location_batches = []
        polarity_batches = []
        document_length_batches = []

        num_batches = len(self.data[1]) // batch_size
        for start, end in zip(
            range(0, (num_batches * batch_size) - batch_size + 1, batch_size),
            range(batch_size, (num_batches * batch_size) + 1, batch_size)
        ):
            text_batch = np.zeros((batch_size, self.max_len_sentences))
            polarity_batch = np.zeros((batch_size, self.emo_nums.max()))
            aspect_location_batch = np.zeros((batch_size, self.max_len_sentences))
            document_lengths = np.zeros(batch_size)

            for document_index, (document, aspect_markup, polarity) in enumerate(self.data[1][start:end]):
                document_length = len(document)
                document_lengths[document_index] = max(document_length, self.max_len_sentences)
                polarity_batch[document_index, polarity] = 1

                if document_length <= self.max_len_sentences:
                    text_batch[document_index, :document_length] = document
                    aspect_location_batch[document_index, :document_length] = aspect_markup
                else:
                    text_batch[document_index] = document[:self.max_len_sentences]
                    aspect_location_batch[document_index] = aspect_markup[:self.max_len_sentences]

            text_batches.append(text_batch)
            polarity_batches.append(polarity_batch)
            aspect_location_batches.append(aspect_location_batch)
            document_length_batches.append(document_lengths)

        return text_batches, aspect_location_batches, polarity_batches, document_length_batches

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

    def __iter__(self):
        for entry in self.data:
            yield entry

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "<Datalist with {} rows>".format(len(self.data))

    @property
    def numberers(self):
        return self.char_nums, self.word_nums, self.emo_nums

    @property
    def n_chars(self):
        return self.char_nums.max()

    @property
    def n_words(self):
        return self.word_nums.max()

    @property
    def n_labels(self):
        return self.emo_nums.max()


class Numberer:

    def __init__(self, first_element=None):
        self.unkown_idx = 0
        if first_element:
            self.num2value = {1 : first_element}
            self.value2num = {first_element : 1}
            self.idx = 2
        else:
            self.num2value = {}
            self.value2num = {}
            self.idx = 1

    def number(self, value, train):

        if train and value not in self.value2num:
            self.value2num[value] = self.idx
            self.num2value[self.idx] = value
            self.idx += 1

        return self.value2num.get(value, self.unkown_idx)

    def __getitem__(self, item):
        return self.value2num[item]

    def max(self):
        return self.idx

    def value(self, num):
        return self.num2value.get(num, None)


if __name__ == '__main__':
    #Testrun
    datalist = Datalist()
    datalist.load_iob("../../../data/train_v1.4.xml", verbose=True)
    batches_iob = datalist.create_iob_batches(512)
    batches_polarity = datalist.create_aspect_polarity_batches(512)
    print(len(batches_iob[0]), len(batches_polarity[0]))
