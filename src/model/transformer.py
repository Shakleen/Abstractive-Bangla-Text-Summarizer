import tensorflow as tf
from .modules.encoder import Encoder
from .modules.decoder import Decoder


train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')


class Transformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        pe_input,
        pe_target,
        rate=0.1
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            pe_input,
            rate
        )

        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            pe_target,
            rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self._loss_object = None

    def call(self, data, training=False):
        inp, tar = data
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self._create_masks(
            inp,
            tar
        )
        enc_output = self.encoder(inp, enc_padding_mask, training)

        dec_output, attention_weights = self.decoder(
            tar,
            enc_output,
            look_ahead_mask,
            dec_padding_mask,
            training,
        )

        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

    def _create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def _create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def _create_masks(self, inp, tar):
        enc_padding_mask = self._create_padding_mask(inp)
        dec_padding_mask = self._create_padding_mask(inp)

        look_ahead_mask = self._create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self._create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def train_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self((inp, tar_inp), training=True)
            loss = self._loss_function(tar_real, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        train_loss(loss)
        train_accuracy(self._accuracy_function(tar_real, predictions))

        return {
            train_loss.name: train_loss.result(),
            train_accuracy.name: train_accuracy.result(),
        }

    def _loss_function(self, real, pred):
        if self._loss_object is None:
            self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction='none'
            )

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self._loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def _accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    @property
    def metrics(self):
        return [train_loss, train_accuracy]

    def test_step(self, data):
        inp, tar = data
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        predictions, _ = self((inp, tar_inp), training=False)
        loss = self._loss_function(tar_real, predictions)

        test_loss(loss)
        test_accuracy(self._accuracy_function(tar_real, predictions))
        
        return {
            test_loss.name: test_loss.result(),
            test_accuracy.name: test_accuracy.result(),
        }
