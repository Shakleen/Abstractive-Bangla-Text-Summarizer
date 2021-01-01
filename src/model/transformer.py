import tensorflow as tf
from .modules.encoder import Encoder
from .modules.decoder import Decoder
from .create_mask import create_masks


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

    def call(
        self,
        inp,
        tar,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask,
        training=False,
    ):
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
