# This custom Scheduler adjusts learning rate over the time
# for the Adam optimizer changes

import tensorflow as tf

# subclass for creating custom learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):                            # the first warmup_steps would grow linearly with every epoch
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)                              #

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)