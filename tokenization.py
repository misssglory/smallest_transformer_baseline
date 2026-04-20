# tokenization.py

import os
import re
import pathlib
import tensorflow as tf
import tensorflow_text as text
from loguru import logger
from tqdm.auto import tqdm
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

from config import (
    QUESTION_VOCAB_PATH,
    ANSWER_VOCAB_PATH,
    VOCAB_SIZE,
    RESERVED_TOKENS,
    BERT_TOKENIZER_PARAMS,
    MAX_TOKENS,
    TOKENIZERS_SAVEDMODEL_DIR,
)
from data_pipeline import normalize_text


bert_vocab_args = dict(
    vocab_size=VOCAB_SIZE,
    reserved_tokens=RESERVED_TOKENS,
    bert_tokenizer_params=BERT_TOKENIZER_PARAMS,
    learn_params={},
)


def write_vocab_file(filepath, vocab):
    filepath = pathlib.Path(filepath)
    with filepath.open("w", encoding="utf-8") as f:
        for token in vocab:
            f.write(token + "\n")


def read_vocab_file(filepath):
    filepath = pathlib.Path(filepath)
    with filepath.open(encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_vocab_with_tqdm(train_ds, which="question"):
    texts = []
    logger.info(f"Копируем все {which} для построения vocab с tqdm...")

    for batch in tqdm(train_ds.batch(1024), desc=f"Collect {which} text"):
        for t in batch.numpy():
            texts.append(t.decode("utf-8"))

    logger.info(
        f"Собрано {len(texts)} строк для {which}-vocab. Стартуем bert_vocab_from_dataset..."
    )

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(texts, dtype=tf.string))
    vocab = bert_vocab(
        ds.batch(1000).prefetch(2),
        **bert_vocab_args,
    )
    return vocab


def maybe_build_or_load_vocab(train_examples):
    if QUESTION_VOCAB_PATH.exists() and ANSWER_VOCAB_PATH.exists():
        logger.info("Найдены существующие vocab-файлы, пропускаем обучение vocab.")
        question_vocab = read_vocab_file(QUESTION_VOCAB_PATH)
        answer_vocab = read_vocab_file(ANSWER_VOCAB_PATH)
    else:
        logger.info("Vocab-файлы не найдены, строим с нуля на всём train.")

        train_questions = train_examples.map(lambda q, a: q)
        train_answers = train_examples.map(lambda q, a: a)

        question_vocab = build_vocab_with_tqdm(train_questions, which="question")
        answer_vocab = build_vocab_with_tqdm(train_answers, which="answer")

        write_vocab_file(QUESTION_VOCAB_PATH, question_vocab)
        write_vocab_file(ANSWER_VOCAB_PATH, answer_vocab)
        logger.info("Словари сохранены на диск.")

    return question_vocab, answer_vocab


def add_start_end(ragged, start_id, end_id):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], start_id)
    ends = tf.fill([count, 1], end_id)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    result = tf.strings.reduce_join(result, separator=" ", axis=-1)
    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path, start_id, end_id):
        self.tokenizer = text.BertTokenizer(str(vocab_path), **BERT_TOKENIZER_PARAMS)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(str(vocab_path))
        self._start_id = tf.constant(start_id, dtype=tf.int64)
        self._end_id = tf.constant(end_id, dtype=tf.int64)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)
        self._vocab_size = len(vocab)

        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string)
        )

        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64)
        )
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64)
        )

        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()
        self.get_start_end_ids.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc, self._start_id, self._end_id)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.constant(self._vocab_size, dtype=tf.int32)

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

    @tf.function
    def get_start_end_ids(self):
        return self._start_id, self._end_id


def create_tokenizers(train_examples):
    question_vocab, answer_vocab = maybe_build_or_load_vocab(train_examples)

    question_tokenizer = text.BertTokenizer(
        str(QUESTION_VOCAB_PATH), **BERT_TOKENIZER_PARAMS
    )
    START = tf.argmax(tf.constant(RESERVED_TOKENS) == "[START]")
    END = tf.argmax(tf.constant(RESERVED_TOKENS) == "[END]")

    tokenizers = tf.Module()
    tokenizers.question = CustomTokenizer(
        RESERVED_TOKENS, QUESTION_VOCAB_PATH, START, END
    )
    tokenizers.answer = CustomTokenizer(RESERVED_TOKENS, ANSWER_VOCAB_PATH, START, END)

    logger.info(f"Сохраняем токенизаторы в SavedModel: {TOKENIZERS_SAVEDMODEL_DIR}")
    tf.saved_model.save(tokenizers, str(TOKENIZERS_SAVEDMODEL_DIR))

    return tokenizers


def load_tokenizers():
    logger.info(f"Загружаем токенизаторы из {TOKENIZERS_SAVEDMODEL_DIR}")
    return tf.saved_model.load(str(TOKENIZERS_SAVEDMODEL_DIR))
