from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

from data import TranslationData

class Model:

    def __init__(self, learning_rate=0.005):
        self.data = TranslationData()
        self.data.load()
        self.learning_rate = learning_rate

    def _simple_gru(self, input_shape):
        output_dim_size = len(self.data.fr_tk.word_index)
        model = Sequential()
        model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
        model.add(TimeDistributed(Dense(1024, activation='relu')))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(output_dim_size + 1, activation='softmax')))

        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(self.learning_rate),
                      metrics=['accuracy'])
        return model

    def build(self, input_shape):
        _model = self._simple_gru(input_shape=input_shape)
        return _model

    def test(self):
        pass

    def run(self):
        _model = self.build(self.data.en_sentences.shape)
        print(_model.summary())
        _model.fit(x=self.data.en_sentences,
                   y=self.data.fr_sentences,
                   batch_size=1024,
                   epochs=5,
                   validation_split=0.3)
        print(self.data.logits_to_text(model.predict(self.data.en_sentences[:1])[0]))

if __name__ == "__main__":
    model = Model()
    model.run()



