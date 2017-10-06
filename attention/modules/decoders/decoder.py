from ..decoders import DecoderBlock
from ...modules import PositionalEmbedding
import sonnet as snt
import tensorflow as tf


class Decoder(snt.AbstractModule):
    def __init__(
            self,
            params,
            block_params,
            embed_params):
        super(Decoder, self).__init__(name="decoder")
        self.params = params
        self.block_params = block_params
        self.embed_params = embed_params

    def _build(self, inputs, sequence_length, encoder_output, encoder_sequence_length, embedding_lookup,
               output_projection=None):

        output = embedding_lookup(inputs)

        output = tf.layers.dropout(output, self.params.dropout_rate)

        for _ in range(self.params.num_blocks):
            output = DecoderBlock(**self.block_params)(output, sequence_length,
                                                       encoder_output, encoder_sequence_length)
        if output_projection is None:
            output_projection = lambda inp: tf.contrib.layers.fully_connected(inp, self.embed_params.vocab_size)

        output = output_projection(output)

        return output
