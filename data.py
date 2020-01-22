from os.path import join as pjoin
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

    def tokenize(self, sentences):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        return tokenizer.texts_to_sequences(sentences), tokenizer

    def _read_file(self, file_name):
        with open(pjoin("data", file_name), "r") as fin:
            lines = fin.readlines()
        return lines

    def load(self):
        en = self._read_file("small_vocab_en")
        fr = self._read_file("small_vocab_fr")
        preprocess_en, en_tk = self.tokenize(en)
        preprocess_fr, fr_tk = self.tokenize(fr)
        preprocess_en = pad_sequences(preprocess_en, padding='post')
        preprocess_fr = pad_sequences(preprocess_fr, padding='post')

        self.logger.info("Max English sentence length {}".format(preprocess_en.shape[1]))
        self.logger.info("Max French sentence length {}".format(preprocess_fr.shape[1]))
        self.logger.info("English vocabulary size {}".format(len(en_tk.word_index)))
        self.logger.info("French vocabulary size {}".format(len(fr_tk.word_index)))


if __name__ == "__main__":
    # data = DataIMDB(top_words=50)
    # data.load()
    data = TranslationData()
    data.load()