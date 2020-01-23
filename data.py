from os.path import join as pjoin
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from misc import get_logger

class DataIMDB:

    def __init__(self, top_words=5000, max_len=256):
        self.top_words = top_words
        self.max_len = max_len

    def load(self):
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=self.top_words,
                                                              maxlen=self.max_len)
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_len)
        print(X_train)
        print(y_train)

        embedding_vecor_length = 32
        model = Sequential()
        model.add(Embedding(50, embedding_vecor_length, input_length=256))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


class TranslationData:

    def __init__(self):
        self.logger = get_logger()
        self.en_tk = None
        self.fr_tk = None
        self.en_sentences = None
        self.fr_sentences = None

    def tokenize(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        return tokenizer.texts_to_sequences(sentences), tokenizer

    def _read_file(self, file_name):
        self.logger.info("START LOAD {}".format(file_name))
        with open(pjoin("data", file_name), "r", encoding="utf8") as fin:
            lines = fin.readlines()
        return lines

    def load(self):
        en = self._read_file("small_vocab_en")
        preprocess_en, en_tk = self.tokenize(en)
        self.en_tk = en_tk
        self.logger.info("English vocabulary size {}".format(len(en_tk.word_index)))
        fr = self._read_file("small_vocab_fr")
        preprocess_fr, fr_tk = self.tokenize(fr)
        self.fr_tk = fr_tk
        self.logger.info("French vocabulary size {}".format(len(fr_tk.word_index)))

        maxlen = max(map(len, preprocess_en + preprocess_fr))
        self.en_sentences = pad_sequences(preprocess_en, maxlen=maxlen, padding='post')
        self.fr_sentences = pad_sequences(preprocess_fr, maxlen=maxlen, padding='post')
        self.en_sentences = self.en_sentences.reshape(*self.en_sentences.shape, 1)
        self.fr_sentences = self.fr_sentences.reshape(*self.fr_sentences.shape, 1)
        assert self.en_sentences.shape == self.fr_sentences.shape

    def logits_to_text(self, logits, mode="to"):
        if mode == "to":
            index_to_words = {i: word for word, i in self.fr_tk.word_index.items()}
        else:
            index_to_words = {i: word for word, i in self.en_tk.word_index.items()}
        index_to_words[0] = '<PAD>'
        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

if __name__ == "__main__":
    # data = DataIMDB(top_words=50)
    # data.load()
    data = TranslationData()
    data.load()