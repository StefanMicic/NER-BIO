import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TokenAndPositionEmbedding(layers.Layer):
    """Class for preparing input to have more context, apart from simple embedding (position)."""

    def __init__(self, max_len: int, vocab_size: int, embed_dim: int) -> None:
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        max_len = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings
