import tensorflow as tf


def _get_dense_layer(size, name):
    return tf.keras.layers.Dense(
        size,
        kernel_initializer=tf.keras.initializers.glorot_normal(),
        name=name
    )


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = _get_dense_layer(d_model, "query")
        self.wk = _get_dense_layer(d_model, "key")
        self.wv = _get_dense_layer(d_model, "value")

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q, mask, training=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q, training=training)
        k = self.wk(k, training=training)
        v = self.wv(v, training=training)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )
        output = self.dense(concat_attention, training=training)

        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scaling operation is performed for faster convergence
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)
        return output, attention_weights
