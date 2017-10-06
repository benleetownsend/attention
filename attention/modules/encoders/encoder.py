import sonnet as snt
from ...modules import PositionalEmbedding

from ..encoders import EncoderBlock
import tensorflow as tf


class Encoder(snt.AbstractModule):
    def __init__(self, params, block_params, embed_params):
        super(Encoder, self).__init__(name="encoder")
        self.params = params
        self.block_params = block_params
        self.embed_params = embed_params

    def _build(self, inputs, sequences_length, reuse_embeddings=True):
        positional_embedding = PositionalEmbedding(**self.embed_params)
        output = positional_embedding(inputs)


        if self.params.dropout_rate > 0.0:
            output = tf.layers.dropout(output, self.params.dropout_rate)
        for _ in range(self.params.num_blocks):
            encoder_block = EncoderBlock(**self.block_params)
            output = encoder_block(output, sequences_length)

        if reuse_embeddings:
            return output, positional_embedding
        return output, None
