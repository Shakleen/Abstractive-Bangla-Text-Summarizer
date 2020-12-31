import tensorflow as tf

def create_tfrecord_dataset(
    tfrecord_files,
    batch_size,
    cache_buffer_size,
    prefetch_buffer_size,
    input_feature_length,
    output_feature_length,
):
  dataset = tf.data.TFRecordDataset(filenames = [tfrecord_files])

  # Create a dictionary describing the features.
  feature_description = {
      'input': tf.io.FixedLenFeature([input_feature_length], tf.int64),
      'target': tf.io.FixedLenFeature([output_feature_length], tf.int64),
  }

  def _parse(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    input, target = example["input"], example["target"]
    return (input, target)

  dataset = dataset.map(_parse)
  dataset = dataset.cache().shuffle(cache_buffer_size).batch(batch_size)
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
  return dataset