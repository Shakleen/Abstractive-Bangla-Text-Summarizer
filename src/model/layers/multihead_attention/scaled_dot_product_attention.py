import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def call(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scaling operation is performed for faster convergence
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        return output, attention_weights
