import tensorflow as tf
import datetime
import os


class LoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, ckpt_manager):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_dir,  current_time + '/train')
        test_log_dir = os.path.join(log_dir, current_time + '/test')
        self.train_summary_writer = tf.summary.create_file_writer(
            train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        self.ckpt_manager = ckpt_manager
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if batch % 50 is 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar(
                    'per_50_batch_loss',
                    logs["train_loss"],
                    step=batch,
                )
                tf.summary.scalar(
                    'per_50_batch_accuracy',
                    logs["train_accuracy"],
                    step=batch,
                )

        return super().on_batch_end(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        # Save every epoch
        self.ckpt_manager.save()
        self._epoch = epoch

        # Log train loss and acc
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                'loss',
                logs["train_loss"],
                step=epoch,
            )
            tf.summary.scalar(
                'accuracy',
                logs["train_accuracy"],
                step=epoch,
            )

        return super().on_epoch_end(epoch, logs=logs)

    def on_test_end(self, logs=None):
        with self.test_summary_writer.as_default():
            tf.summary.scalar(
                'loss',
                logs["test_loss"],
                step=self._epoch,
            )
            tf.summary.scalar(
                'accuracy',
                logs["test_accuracy"],
                step=self._epoch,
            )
        return super().on_test_end(logs=logs)
