from os.path import join as pjoin
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy
import random
import math
import time

from misc import get_logger


SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _TranslationDataset(Dataset):

    def __init__(self):
        self.logger = get_logger()
        self.en_tk = None
        self.fr_tk = None
        self.en_sentences = None
        self.fr_sentences = None

    def _read_file(self, file_name):
        self.logger.info("START LOAD {}".format(file_name))
        with open(pjoin("data", file_name), "r", encoding="utf8") as fin:
            lines = fin.readlines()
        return lines

    def _tokenize(self, sentences):
        word_dict = dict()
        pivot = 1
        sequences = []
        for sentence in sentences:
            words = sentence.strip().split()
            for word in words:
                if word not in word_dict:
                    word_dict[word] = pivot
                    pivot = pivot + 1
            tokens = [word_dict[w] for w in words]
            sequences.append(tokens)
        return sequences, word_dict



class Data:

    def __init__(self):
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')
        self.source = Field(tokenize=lambda text: [t.text for t in spacy_de.tokenizer(text)][::-1],
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
        self.target = Field(tokenize=lambda text: [t.text for t in spacy_en.tokenizer(text)],
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

    def load(self):
        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(self.source, self.target))
        print("Number of training examples: {}".format(len(train_data.examples)))
        print("Number of validation examples: {}".format(len(valid_data.examples)))
        print("Number of testing examples: {}".format(len(test_data.examples)))

        self.source.build_vocab(train_data, min_freq=2)
        self.target.build_vocab(train_data, min_freq=2)
        print("Unique tokens in source (de) vocabulary: {}".format(len(self.source.vocab)))
        print("Unique tokens in source (en) vocabulary: {}".format(len(self.target.vocab)))
        return train_data, valid_data, test_data

    def iterator(self):
        train_data, valid_data, test_data = self.load()
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = 128,
            device=device)
        return train_iter, valid_iter, test_iter

if __name__ == "__main__":
    data = Data()
    data.load()
