import tensorflow as tf

class PointwiseFeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(dff, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x