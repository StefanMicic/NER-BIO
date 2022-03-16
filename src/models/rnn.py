import numpy as np
from keras.layers import (
    Bidirectional, Dense,
    Embedding,
    Input, LSTM,
)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from data_preprocessing.word2vec import train_word2vec
from utils import split_features_and_labels


class RNNModel:
    """Class for bi-directional RNN model architecture and data preprocessing."""

    def __init__(self, cfg):
        self.y = None
        self.data = None
        self.max_len = None
        self.cfg = cfg
        self.tokenizer = Tokenizer(num_words=self.cfg['training']['vocab_size'])
        self.word2idx = None
        self.num_words = 0

    def preprocessing(self) -> None:
        input_data_path = self.cfg['data_preprocessing']['input_data_path']
        train_data = np.load(f'{input_data_path}/train.pkl', allow_pickle=True)
        _, labels = split_features_and_labels(train_data)
        text = []
        for record in train_data:
            text.append([x.lower() for x in record[0]])

        self.tokenizer.fit_on_texts(text[:self.cfg['data_preprocessing']['num_of_samples']])
        sequences = self.tokenizer.texts_to_sequences(text[:50])
        self.word2idx = self.tokenizer.word_index
        self.num_words = min(self.cfg['training']['vocab_size'], len(self.word2idx) + 1)
        self.max_len = max([len(s) for s in text[:self.cfg['data_preprocessing']['num_of_samples']]])
        self.data = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        mapping = {key: value for value, key in enumerate(labels)}
        y_data = []

        for record in train_data[:self.cfg['data_preprocessing']['num_of_samples']]:
            y_data.append(np.asarray([mapping[x] for x in record[1]]))
        self.y = np.asarray(pad_sequences(y_data, maxlen=self.max_len, dtype='int32', padding='post',
                                          value=63))

    def create_embed_matrix(self) -> np.array:
        word2vec = train_word2vec(self.cfg)
        embedding_matrix = np.zeros((self.num_words, self.cfg['training']['embed_dim']))
        for word, i in self.word2idx.items():
            if i < self.cfg['training']['vocab_size']:
                try:
                    embedding_vector = word2vec.wv[word]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                except:
                    pass
        return embedding_matrix

    def create_model(self, embedding_matrix: np.array) -> Model:
        embedding_layer = Embedding(
            self.num_words,
            self.cfg['training']['embed_dim'],
            weights=[embedding_matrix],
            input_length=self.max_len,
            trainable=True,
        )
        input_ = Input(shape=(self.max_len,))
        x = embedding_layer(input_)
        x = Bidirectional(LSTM(15, return_sequences=True))(x)
        output = Dense(self.cfg['training']['num_of_labels'] + 1, activation="softmax")(x)

        return Model(input_, output)

    def train_rnn(self, model: Model) -> Model:
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=Adam(lr=0.01),
            metrics=["accuracy"],
        )
        print(self.data[:10])
        model.fit(
            self.data,
            self.y,
            batch_size=self.cfg['training']['batch_size'],
            epochs=self.cfg['training']['epochs'],
            validation_split=0.2,
        )
        return model
