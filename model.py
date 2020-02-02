import fire
import pathlib
import numpy as np
from os.path import join as pjoin

from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard


from misc import get_logger
from data import TranslationData


class TranslationModel:
    """
    1. rnn, lstm, gru, bdlstm, attention
    """

    def __init__(self, learning_rate, batch_size, epochs, validation_rate):
        self.data = TranslationData()
        self.data.load()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_rate = validation_rate
        self._feed_forward = None

    def build(self):
        raise NotImplementedError

    def feed_forward(self, x):
        if self._feed_forward is not None:
            raise NotImplementedError
        inference = self._feed_forward([x.reshape(1, -1, 1), 0])
        return inference.reshape(-1) # flatten

    def feed_forward_batch(self, x):
        if self._feed_forward is not None:
            raise NotImplementedError
        inference = self._feed_forward([x, 0])
        return inference

    def run(self):
        net = self.build()
        print(net.summary())
        net.fit(x=self.data.en_sentences,
                y=self.data.fr_sentences,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=self.validation_rate)
        print(self.data.logits_to_text(net.predict(self.data.en_sentences[:1])[0]))


class GRU(Model):

    def build(self):
        input_shape = self.data.en_sentences.shape
        output_dim_size = len(self.data.fr_tk.word_index)
        model = Sequential()
        model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
        model.add(TimeDistributed(Dense(1024, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(output_dim_size + 1, activation='softmax')))

        # model = Model()
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(self.learning_rate),
                      metrics=['accuracy'])
        self._feed_forward = K.function([model.input, K.learning_phase()],
                                        [model.get_layer("logits").output])
        return model

    def test(self):
        pass


if __name__ == "__main__":
    model = Model()
    model.run()
