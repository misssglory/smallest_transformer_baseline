# data_pipeline.py

import tensorflow as tf
from datasets import load_dataset
from loguru import logger

from config import HF_DATASET_NAME, HF_SPLIT, HF_TEST_SIZE, HF_SEED


def normalize_text(x):
    if x is None:
        return ""
    return str(x).strip()


def example_generator(hf_ds):
    for row in hf_ds:
        q = normalize_text(row.get("question"))
        a = normalize_text(row.get("answer"))
        if not q or not a:
            continue
        yield q, a


def load_hf_splits():
    logger.info(f"Загружаем датасет {HF_DATASET_NAME} ({HF_SPLIT})...")
    dataset = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    dataset = dataset.train_test_split(test_size=HF_TEST_SIZE, seed=HF_SEED)

    train_hf = dataset["train"]
    val_hf = dataset["test"]

    logger.info(f"Размер train HF: {len(train_hf)}")
    logger.info(f"Размер val   HF: {len(val_hf)}")
    return train_hf, val_hf


def make_tf_datasets(train_hf, val_hf):
    train_examples = tf.data.Dataset.from_generator(
        lambda: example_generator(train_hf),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    val_examples = tf.data.Dataset.from_generator(
        lambda: example_generator(val_hf),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        ),
    )

    return train_examples, val_examples
