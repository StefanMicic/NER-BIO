import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.embedding import TokenAndPositionEmbedding
from models.transformer_block import TransformerBlock


class NERModel(keras.Model):
    """Class for NER model architecture with embedding and transformer layer."""

    def __init__(
            self, num_tags: int, vocab_size: int, maxlen: int, embed_dim: int = 32, num_heads: int = 2, ff_dim: int = 32
    ):
        super(NERModel, self).__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x
