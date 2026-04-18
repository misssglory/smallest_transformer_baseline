#!/usr/bin/env python
# coding: utf-8

# ============================================
# Кастомный Transformer для question -> answer
# Датасет: Den4ikAI/russian_dialogues
# Минимальные изменения от оригинального кода
# ============================================

# Если вы запускаете в Colab/Jupyter и нужны совместимые версии:
# !apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
# !pip uninstall -y -q tensorflow keras tensorflow-estimator tensorflow-text
# !pip install protobuf~=3.20.3
# !pip install -q -U tensorflow-text tensorflow datasets

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text as text
import re
import pathlib
import warnings

from loguru import logger
from datasets import load_dataset
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

warnings.filterwarnings("ignore")

# ============================================
# 1. Загрузка датасета
# ============================================

logger.info("Загружаем датасет Den4ikAI/russian_dialogues...")
dataset = load_dataset("Den4ikAI/russian_dialogues", split="train")
dataset = dataset.train_test_split(test_size=0.01, seed=42)

train_hf = dataset["train"]
val_hf = dataset["test"]

logger.info(f"Размер обучающей выборки: {len(train_hf)}")
logger.info(f"Размер валидационной выборки: {len(val_hf)}")


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

for q_examples, a_examples in train_examples.batch(3).take(1):
    logger.info("Примеры вопросов:")
    for q in q_examples.numpy():
        logger.info(q.decode("utf-8"))
    logger.info("Примеры ответов:")
    for a in a_examples.numpy():
        logger.info(a.decode("utf-8"))

# ============================================
# 2. Токенизация
# ============================================

VOCAB_SIZE = 8000
bert_tokenizer_params = dict(lower_case=True)
reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]

bert_vocab_args = dict(
    vocab_size=VOCAB_SIZE,
    reserved_tokens=reserved_tokens,
    bert_tokenizer_params=bert_tokenizer_params,
    learn_params={},
)

# Для минимальных изменений сохраняем две ветки: "question" и "answer"
train_questions = train_examples.map(lambda q, a: q)
train_answers = train_examples.map(lambda q, a: a)

logger.info("Строим словарь вопросов...")
question_vocab = bert_vocab.bert_vocab_from_dataset(
    train_questions.batch(1000).prefetch(2), **bert_vocab_args
)

logger.info("Строим словарь ответов...")
answer_vocab = bert_vocab.bert_vocab_from_dataset(
    train_answers.batch(1000).prefetch(2), **bert_vocab_args
)


def write_vocab_file(filepath, vocab):
    with open(filepath, "w", encoding="utf-8") as f:
        for token in vocab:
            print(token, file=f)


write_vocab_file("question_vocab.txt", question_vocab)
write_vocab_file("answer_vocab.txt", answer_vocab)

logger.info("Фрагменты question_vocab:")
logger.info(str(question_vocab[:20]))
logger.info(str(question_vocab[100:120]))

logger.info("Фрагменты answer_vocab:")
logger.info(str(answer_vocab[:20]))
logger.info(str(answer_vocab[100:120]))

question_tokenizer = text.BertTokenizer("question_vocab.txt", **bert_tokenizer_params)
answer_tokenizer = text.BertTokenizer("answer_vocab.txt", **bert_tokenizer_params)

# Посмотрим длины
lengths = []
for q_examples, a_examples in train_examples.batch(1024).take(200):
    q_tokens = question_tokenizer.tokenize(q_examples)
    lengths.append(q_tokens.row_lengths())
    a_tokens = answer_tokenizer.tokenize(a_examples)
    lengths.append(a_tokens.row_lengths())

all_lengths = np.concatenate([x.numpy() for x in lengths])

plt.hist(all_lengths, np.linspace(0, 100, 101))
plt.ylim(plt.ylim())
max_length = max(all_lengths)
plt.plot([max_length, max_length], plt.ylim())
plt.title(f"Максимальное количество токенов в примере: {max_length}")
plt.show()

# Для коротких диалоговых фраз разумно взять 40-64
MAX_TOKENS = 40

START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")


def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count, 1], START)
    ends = tf.fill([count, 1], END)
    return tf.concat([starts, ragged, ends], axis=1)


def cleanup_text(reserved_tokens, token_txt):
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    result = tf.strings.reduce_join(result, separator=" ", axis=-1)
    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = tf.Variable(vocab)

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

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc)
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
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


tokenizers = tf.Module()
tokenizers.question = CustomTokenizer(reserved_tokens, "question_vocab.txt")
tokenizers.answer = CustomTokenizer(reserved_tokens, "answer_vocab.txt")

model_name = "russian_dialogues_qa_converter"
logger.info(f"Сохраняем токенизаторы в SavedModel: {model_name}")
tf.saved_model.save(tokenizers, model_name)

# Проверка токенизации
for q_examples, a_examples in train_examples.batch(3).take(1):
    encoded = tokenizers.question.tokenize(q_examples)
    logger.info("Токенизация вопросов:")
    for row in encoded.to_list():
        logger.info(str(row))

    round_trip = tokenizers.question.detokenize(encoded)
    logger.info("Обратная детокенизация вопросов:")
    for line in round_trip.numpy():
        logger.info(line.decode("utf-8"))

# ============================================
# 3. Подготовка батчей
# ============================================


def prepare_batch(question, answer):
    question = tokenizers.question.tokenize(question)
    question = question[:, :MAX_TOKENS]
    question = question.to_tensor()

    answer = tokenizers.answer.tokenize(answer)
    answer = answer[:, : (MAX_TOKENS + 1)]

    answer_inputs = answer[:, :-1].to_tensor()
    answer_labels = answer[:, 1:].to_tensor()

    return (question, answer_inputs), answer_labels


BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_batches(ds):
    return (
        ds.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (question, answer), answer_labels in train_batches.take(1):
    break

logger.info("Формы батча:")
logger.info(f"question: {question.shape}")
logger.info(f"answer: {answer.shape}")
logger.info(f"answer_labels: {answer_labels.shape}")

# ============================================
# 4. Архитектура модели
# ============================================


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x, key=context, value=context, return_attention_scores=True
        )

        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1
    ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dec_layers = [
            DecoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


class Transformer(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        target_vocab_size,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=input_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=target_vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs

        context = self.encoder(context)
        x = self.decoder(x, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


# ============================================
# 5. Обучение
# ============================================

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
EPOCHS = 10

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.question.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.answer.get_vocab_size().numpy(),
    dropout_rate=dropout_rate,
)

# Проверка одного forward pass
output = transformer((question, answer))
logger.info("Проверка трансформера:")
logger.info(f"answer: {answer.shape}")
logger.info(f"question: {question.shape}")
logger.info(f"output: {output.shape}")

attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
logger.info(f"attn_scores.shape: {attn_scores.shape}")


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        step = tf.maximum(step, 1.0)

        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)

    match = label == pred
    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)


transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])

logger.info("Запускаем обучение...")
history = transformer.fit(train_batches, epochs=EPOCHS, validation_data=val_batches)

# ============================================
# 6. Inference
# ============================================


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)

        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tokenizers.question.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.answer.tokenize([""])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]
            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i + 1, predicted_id[0])

            if tf.reduce_all(tf.equal(predicted_id, end)):
                break

        output = tf.transpose(output_array.stack())

        text = self.tokenizers.answer.detokenize(output)[0]
        tokens = self.tokenizers.answer.lookup(output)[0]

        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores

        return text, tokens, attention_weights


translator = Translator(tokenizers, transformer)


def print_answer(question, predicted_answer, ground_truth):
    logger.info(f'{"Вопрос":25s}: {question}')
    logger.info(
        f'{"Предсказанный ответ":25s}: {predicted_answer.numpy().decode("utf-8")}'
    )
    logger.info(f'{"Оригинальный ответ":25s}: {ground_truth}')


bad_q = 0
bad_a = 0
bad_empty = 0

for row in train_hf.select(range(min(200000, len(train_hf)))):
    q = row.get("question")
    a = row.get("answer")

    if q is None:
        bad_q += 1
    if a is None:
        bad_a += 1
    if not (q or "").strip() or not (a or "").strip():
        bad_empty += 1

logger.info(f"bad_q = {bad_q}")
logger.info(f"bad_a = {bad_a}")
logger.info(f"bad_empty = {bad_empty}")

# Несколько примеров из датасета
samples = train_hf.select(range(5))

for sample in samples:
    q = sample["question"]
    gt = sample["answer"]

    predicted_text, predicted_tokens, attention_weights = translator(tf.constant(q))
    print_answer(q, predicted_text, gt)
    logger.info("-" * 80)

# ============================================
# 7. Визуализация внимания
# ============================================


def plot_attention_head(in_tokens, translated_tokens, attention):
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)

    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode("utf-8") for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode("utf-8") for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)


def plot_attention_weights(sentence, translated_tokens, attention_heads):
    in_tokens = tf.convert_to_tensor([sentence])
    in_tokens = tokenizers.question.tokenize(in_tokens).to_tensor()
    in_tokens = tokenizers.question.lookup(in_tokens)[0]

    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h + 1)
        plot_attention_head(in_tokens, translated_tokens, head)
        ax.set_xlabel(f"Head {h + 1}")

    plt.tight_layout()
    plt.show()


# Пример внимания
sample_question = "как дела?"
predicted_text, predicted_tokens, attention_weights = translator(
    tf.constant(sample_question)
)

logger.info("Пример ответа:")
logger.info(f"Вопрос: {sample_question}")
logger.info(f"Ответ : {predicted_text.numpy().decode('utf-8')}")

plot_attention_weights(sample_question, predicted_tokens, attention_weights[0])

# ============================================
# 8. Экспорт модели
# ============================================


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens, attention_weights = self.translator(
            sentence, max_length=MAX_TOKENS
        )
        return result


export_translator = ExportTranslator(translator)

logger.info("Проверка экспортируемого модуля:")
logger.info(export_translator(tf.constant("как дела?")).numpy().decode("utf-8"))

tf.saved_model.save(export_translator, export_dir="translator")
reloaded = tf.saved_model.load("translator")

logger.info("Проверка после загрузки:")
logger.info(reloaded(tf.constant("как дела?")).numpy().decode("utf-8"))
