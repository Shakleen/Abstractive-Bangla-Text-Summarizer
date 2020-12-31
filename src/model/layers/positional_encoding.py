import tensorflow as tf


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        tf.arange(position)[:, tf.newaxis],
        tf.arange(d_model)[tf.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = tf.math.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[tf.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(position, i, d_model):
    base = 10000
    power = (2 * (i // 2)) / tf.float32(d_model)
    return position / tf.math.pow(x=base, y=power)
