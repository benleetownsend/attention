import sonnet as snt
from ..encoders import Encoder
from ..decoders import Decoder, BeamSearchDecoder
from tensorflow.python.estimator.model_fn import ModeKeys
from attention.modules.core.embedding import PositionalEmbedding


class TransformerModule(snt.AbstractModule):
    def __init__(self, params, mode_key):
        super(TransformerModule, self).__init__(name="transformer")
        self.params = params
        self.mode_key = mode_key

    def _build(self, features, output_projection=None):
        encoder_inputs, encoder_length = features[0]

        encoder = Encoder(
            params=self.params.encoder_params.params,
            block_params=self.params.encoder_params.encoder_block_params,
            embed_params=self.params.encoder_params.embed_params
        )

        encoder_output, positional_embedding = encoder(inputs=encoder_inputs, sequences_length=encoder_length)
        decoder = Decoder(
            params=self.params.decoder_params.params,
            block_params=self.params.decoder_params.decoder_block_params,
            embed_params=self.params.decoder_params.embed_params,
        )

        decoder_embeddings = PositionalEmbedding(**self.params.decoder_params.embed_params)

        if self.mode_key == ModeKeys.PREDICT:
            bs_decoder = BeamSearchDecoder(decoder=decoder)
            logits = bs_decoder(encoder_output,
                                max_sequence_length=self.params.decoder_params.max_sequence_length,
                                beam_size=self.params.decoder_params.beam_size,
                                output_projection=output_projection,
                                embedding_lookup=decoder_embeddings)

        else:
            decoder_inputs, decoder_length = features[1]
            logits = decoder(inputs=decoder_inputs[:, :-1], sequence_length=decoder_length,
                             encoder_output=encoder_output, encoder_sequence_length=encoder_length,
                             embedding_lookup=decoder_embeddings, output_projection=output_projection)
        return logits
