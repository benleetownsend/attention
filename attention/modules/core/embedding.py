import tensorflow as tf
import sonnet as snt


class PositionalEmbedding(snt.AbstractModule):
    def __init__(self, vocab_size, embed_dim, max_sequence_length):
        super(PositionalEmbedding, self).__init__(name="positional_embedding")

        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        with self._enter_variable_scope():
            self.embed = snt.Embed(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                name="embedding"
            )

    def _build(self, ids):
        emb_lookup = self.embed(ids)
        positional_embedding = tf.get_variable('positional_embedding',
                                               dtype=tf.float32,
                                               shape=[self.max_sequence_length, self.embed_dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)

        positional_embedding = positional_embedding[:emb_lookup.shape[1]]

        return emb_lookup + positional_embedding
