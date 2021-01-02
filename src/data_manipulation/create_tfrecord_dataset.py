import tensorflow as tf

def create_tfrecord_dataset(
    tfrecord_files,          # TFRecord file list
    batch_size,              # each batch in the dataset will contain 32 examples by default
    cache_buffer_size,       # 8192 dataset elements will be buffered for shuffling to achieve randomization by default
    prefetch_buffer_size,    # number of elements needs to be buffered will be dynamically tuned
    input_feature_length,    # article size is limited to 512 words by default
    output_feature_length,   # summary size is limited to 12 words by default
):

  # Create a source dataset from your content file list.
  dataset = tf.data.TFRecordDataset(filenames = [tfrecord_files])

  # Create a dictionary of describing the fixed-length features.
  feature_description = {
      'input': tf.io.FixedLenFeature([input_feature_length], tf.int64),        # one dimensional integer list to represent the example content
      'target': tf.io.FixedLenFeature([output_feature_length], tf.int64),      # int64 list is expected as the summary output
  }

  # Define a private parsing function to extract 'input', 'target' features from each example
  def _parse(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)   # 
    input, target = example["input"], example["target"]
    return (input, target)

  # transforms _parse across all the elements on dataset
  dataset = dataset.map(_parse)
  dataset = dataset.cache().shuffle(cache_buffer_size).batch(batch_size)       # caches the dataset to memory in chunks and shuffle the samples
  dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)                 # setting number of batches to prefetch to improve throughput
  return dataset