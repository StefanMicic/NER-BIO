from collections import Counter

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from loguru import logger as log
from tensorflow import keras
from tqdm import tqdm

from utils import lowercase_and_convert_to_ids, split_features_and_labels


class DataPreprocessing:
    """Class for preparing data for training of the transformer model."""

    def __init__(self, train_data: np.array, vocab_size: int):
        self.train_data_text, self.tokens = split_features_and_labels(train_data)
        self.counter = Counter(self.train_data_text)
        self.mapping = {key: value for value, key in enumerate(self.tokens)}
        self.vocab_size = vocab_size
        vocabulary = [token for token, count in self.counter.most_common(self.vocab_size - 2)]
        self.lookup_layer = keras.layers.StringLookup(
            vocabulary=vocabulary
        )

    def __call__(self, data_path: str, data: np.array, num_of_samples: int):
        log.info("Data is being prepared")
        x_data = []
        y_data = []

        for record in tqdm(data[:num_of_samples]):
            x_data.append(np.asarray(lowercase_and_convert_to_ids(record[0], self.lookup_layer)))
            y_data.append(np.asarray([self.mapping[x] for x in record[1]]))

        max_len = max([len(s) for s in x_data])

        x = np.asarray(pad_sequences(x_data, maxlen=max_len, dtype='int32', padding='post',
                                     value=self.vocab_size))
        y = np.asarray(pad_sequences(y_data, maxlen=max_len, dtype='int32', padding='post',
                                     value=self.get_num_tags()))

        np.save(f"{data_path}_x.npy", x)
        np.save(f"{data_path}_y.npy", y)

    def get_num_tags(self):
        return len(self.mapping)
