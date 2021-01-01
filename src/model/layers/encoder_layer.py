import tensorflow as tf
from .multihead_attention import MultiHeadAttention
from .pointwise_feed_forward_network import PointwiseFeedForwardNetwork


def _get_layer_norm():
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PointwiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = _get_layer_norm()
        self.layernorm2 = _get_layer_norm()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training=False):
        attn_output, _ = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
