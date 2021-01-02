"""Python script to process and create tfrecords.

The script reads csv file from *input_csv_dir*. The records are split into train and test set.
*split_ratio* amount of records from each csv_file are used for generating training tfrecords.

Command:
python3 src/data_manipulation/create_tfrecords_from_csv.py \
--input_csv_dir /run/media/ishrak/Ishrak/IUT/Thesis/dataset/csvs/ \
--output_dir /run/media/ishrak/Ishrak/IUT/Thesis/dataset/tfrecords/
"""

import os
import argparse
import time
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from bpemb import BPEmb


def parse_args() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_csv_dir",
        type=str,
        required=True,
        help="path to input csv files"
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )
    ap.add_argument(
        "-vs",
        "--vocab_size",
        type=int,
        default=10000,
        required=False,
        help="Size of vocabulary"
    )
    ap.add_argument(
        "-vd",
        "--vocab_dim",
        type=int,
        default=100,
        required=False,
        help="Dimension of embeddings"
    )
    ap.add_argument(
        "-sl",
        "--summary_length",
        type=int,
        default=16,
        required=False,
        help="Max summary length"
    )
    ap.add_argument(
        "-tl",
        "--text_length",
        type=int,
        default=512,
        required=False,
        help="Max text length"
    )
    ap.add_argument(
        "-sp",
        "--split_ratio",
        type=float,
        default=0.90,
        required=False,
        help="Percentage of records in the csv file to be used in training"
    )

    return vars(ap.parse_args())


def get_csv_file_paths() -> list:
    csv_files = [
        os.path.join(CSV_DIR, file_name)
        for file_name in os.listdir(CSV_DIR)
        if file_name.split(".")[-1] == "csv"
    ]
    print(f"Total files found: {len(csv_files)}")
    return csv_files


def process_file(csv_file_path):
    try:
        print(f"Processing: {csv_file_path}")
        df = pd.read_csv(csv_file_path, usecols=["title", "content"])
        df = clean_dataframe(df)
        train, test = train_test_split(df)

        file_basename = csv_file_path.split("/")[-1].split(".")[0]
        convert_df(
            train,
            os.path.join(OUTPUT_TRAIN_DIR, f"{file_basename}.tfrecord")
        )
        convert_df(
            test,
            os.path.join(OUTPUT_TEST_DIR, f"{file_basename}.tfrecord")
        )
    except Exception:
        print(f"Encountered an error for file {csv_file_path}")
        print(traceback.format_exc())


def clean_dataframe(df) -> pd.DataFrame:
    df = df.dropna(axis=0, how="any")
    return df


def train_test_split(df):
    df = df.sample(frac=1)
    total_rows = df.shape[0]
    train_rows = int(total_rows * SPLIT_RATIO)
    train = df.iloc[:train_rows, :]
    test = df.iloc[train_rows:, :]
    return train, test


def convert_df(df, output_path):
    inputs, targets = preprocess(df["title"].values, df["content"].values)
    create_tfrecord(inputs, targets, output_path)


def preprocess(summary, text) -> tuple:
    inputs, targets = byte_pair_encode(text, summary)
    inputs = pad_and_truncate(inputs, TEXT_LENGTH)
    targets = pad_and_truncate(targets, SUMMARY_LENGTH)
    inputs = tf.cast(inputs, dtype=tf.int64)
    targets = tf.cast(targets, dtype=tf.int64)
    return inputs, targets


def byte_pair_encode(text, summary):
    inputs = BYTE_PAIR_ENCODER.encode_ids(text)
    targets = BYTE_PAIR_ENCODER.encode_ids_with_bos_eos(summary)
    return inputs, targets


def pad_and_truncate(seq, len):
    return tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=len, padding='post', truncating='post')


def create_tfrecord(inputs, targets, output_path):
    with tf.io.TFRecordWriter(output_path) as writer:
        for (input, target) in zip(inputs, targets):
            tf_example = create_example(input, target)
            writer.write(tf_example.SerializeToString())


def create_example(input, target) -> tf.train.Example:
    feature = {
        'input': _bytes_feature(input),
        'target': _bytes_feature(target),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _bytes_feature(value) -> tf.train.Feature:
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


args = parse_args()
CSV_DIR = args["input_csv_dir"]
OUPTUT_DIR = args["output_dir"]
OUTPUT_TRAIN_DIR = os.path.join(OUPTUT_DIR, "train")
OUTPUT_TEST_DIR = os.path.join(OUPTUT_DIR, "test")
VOCAB_SIZE = args["vocab_size"]
VOCAB_DIM = args["vocab_dim"]
SUMMARY_LENGTH = args["summary_length"]
TEXT_LENGTH = args["text_length"]
SPLIT_RATIO = args["split_ratio"]

BYTE_PAIR_ENCODER = BPEmb(lang="bn", vs=VOCAB_SIZE, dim=VOCAB_DIM)

if not os.path.exists(OUPTUT_DIR):
    os.makedirs(OUPTUT_DIR)
    os.makedirs(OUTPUT_TRAIN_DIR)
    os.makedirs(OUTPUT_TEST_DIR)

csv_files = get_csv_file_paths()

start = time.time()

for csv_file in csv_files:
    process_file(csv_file)

end = time.time()
print(f"FINISHED in {end-start:.4f}")
