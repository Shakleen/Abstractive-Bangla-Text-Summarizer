import tensorflow as tf
from .multihead_attention import MultiHeadAttention
from .positionwise_feed_forward_network import PositionwiseFeedForwardNetwork


def _get_layer_norm():
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = PositionwiseFeedForwardNetwork(d_model, dff)

        self.layernorm1 = _get_layer_norm()
        self.layernorm2 = _get_layer_norm()
        self.layernorm3 = _get_layer_norm()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=False):
        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output,
            enc_output,
            out1,
            padding_mask,
            training=training
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2
