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

class FRENDataset(Dataset):

    def __init__(self):
        self.logger = get_logger()
        self.init_token = '<sos>'
        self.end_token = '<eos>'
        en = self._read_file("small_vocab_en")
        fr = self._read_file("small_vocab_fr")
        assert len(en) == len(fr)
        self.en_sequences, self.en_dict = self._tokenize(en)
        self.fr_sequences, self.fr_dict = self._tokenize(fr)
        self.source_dim = len(self.en_dict) + 1
        self.target_dim = len(self.fr_dict) + 1
        self.end_token_pivot = 2

    def _read_file(self, file_name):
        self.logger.info("START LOAD {}".format(file_name))
        with open(pjoin("data", file_name), "r", encoding="utf8") as fin:
            lines = fin.readlines()
        return lines

    def _tokenize(self, sentences):
        word_dict = {
            self.init_token: 1,
            self.end_token: 2,
        }
        pivot = 3
        sequences = []
        for sentence in sentences:
            words = sentence.strip().split()
            words = [self.init_token] + words + [self.end_token]
            for word in words:
                if word not in word_dict:
                    word_dict[word] = pivot
                    pivot = pivot + 1
            tokens = [word_dict[w] for w in words]
            sequences.append(tokens)
        max_len = max(map(len, sequences))
        self.logger.info("MAX LEN : {}".format(max_len))
        self.logger.info("TOTAL SEQ: {}".format(len(sequences)))
        for sequence in sequences:
            seq_len = len(sequence)
            sequence.extend([2] * (max_len - seq_len))
        return torch.tensor(sequences), word_dict

    def __len__(self):
        return len(self.en_sequences)

    def __getitem__(self, index):
        return self.en_sequences[index], self.fr_sequences[index]



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

