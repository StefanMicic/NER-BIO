from typing import List, Set, Tuple

import numpy as np
import tensorflow as tf
from keras.layers.preprocessing.string_lookup import StringLookup
from tqdm import tqdm


def lowercase_and_convert_to_ids(tokens: List, lookup_layer: StringLookup) -> tf.Tensor:
    tokens = tf.strings.lower(tokens)
    return lookup_layer(tokens)


def split_features_and_labels(train_data: np.array) -> Tuple[List, Set]:
    train_data_text = []
    tokens = set()
    for record in tqdm(train_data):
        train_data_text.extend(record[0])
        tokens.update(record[1])
    return train_data_text, tokens
