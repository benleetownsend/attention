import sonnet as snt
import tensorflow as tf


class BeamSearchDecoder(snt.AbstractModule):
    def __init__(self, decoder):
        super(BeamSearchDecoder, self).__init__(name="bs_decoder")
        self.decoder = decoder
        self.params = decoder.params

    def _build(self, encoder_output, max_sequence_length, beam_size, embedding_lookup=None,
               bs_heuristic=None, output_projection=None):
        encoder_output = tf.concat([encoder_output for _ in range(beam_size)], axis=0)
        sequence_lengths_base = tf.ones_like(encoder_output[:, 0, 0], dtype=tf.int32)
        decoder_input = tf.ones(shape=[beam_size, 1], dtype=tf.int32) * self.params["bos"]
        if bs_heuristic is None:
            bs_heuristic = LogProbHeuristic
        bs = bs_heuristic(beam_size, self.params.eos).standard_bs_heuristic

        for sequence_lengths_offset in range(max_sequence_length):
            seq_lens = sequence_lengths_base + sequence_lengths_offset
            decoder_output = self.decoder(decoder_input, seq_lens, encoder_output, sequence_lengths_base,
                                          embedding_lookup=embedding_lookup, output_projection=output_projection)

            beam_idxs, ids = bs(decoder_output, decoder_input)  # bs_in, vocab_dim
            selected_beams = tf.gather(decoder_input, beam_idxs)
            decoder_input = tf.concat([selected_beams, tf.expand_dims(ids, dim=-1)], axis=-1)

        return decoder_input


class LogProbHeuristic:
    def __init__(self, beam_size, eos):
        self.beam_size = beam_size
        self.log_probs = tf.zeros(shape=[beam_size])
        self.eos = eos

    @staticmethod
    def twod_top_k(input_logprobs, k):
        values, indices = tf.nn.top_k(tf.reshape(input_logprobs, shape=[-1]), k=k)
        ids = tf.mod(indices, input_logprobs.shape[1])
        beams = tf.cast(tf.floordiv(indices, input_logprobs.shape[1]), dtype=tf.int32)
        beams = tf.cast(beams, dtype=tf.int32)
        return ids, beams, values

    def standard_bs_heuristic(self, decoder_output, beams, alpha=0.2):  # takes bs, sequence_lengths, projection_dim
        penalty = ((1 + decoder_output.get_shape().as_list()[-1]) ** alpha) / (6 ** alpha)
        last_tokens = decoder_output[:, -1, :]
        log_probs = tf.nn.log_softmax(last_tokens) / penalty
        if beams.get_shape().as_list()[1] > 1:
            finished_beams = tf.reduce_sum(
                tf.div(tf.cast(tf.not_equal(beams[:, 1:], self.eos), dtype=tf.float32), beams.get_shape().as_list()[1]),
                axis=1)
        else:
            finished_beams = tf.ones([self.beam_size], dtype=tf.float32)

        running_log_probs = tf.transpose((tf.transpose(log_probs) + self.log_probs) * finished_beams)
        ids, beams, values = self.twod_top_k(running_log_probs, self.beam_size)
        self.log_probs = tf.gather(self.log_probs, beams)
        self.log_probs += values
        return beams, ids
