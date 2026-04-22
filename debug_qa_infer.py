# debug_qa_infer.py
#
# Чистый отладочный инференс без SavedModel:
#  - грузим HF-данные и токенизаторы
#  - создаём / загружаем Transformer
#  - запускаем greedy-декодер
#  - печатаем сырые токены и detokenize

import tensorflow as tf
import tensorflow_text as text  # noqa: F401
from loguru import logger

from config import MAX_TOKENS, LOG_LEVEL
from data_pipeline import load_hf_splits, make_tf_datasets
from tokenization import load_tokenizers
from model import create_or_load_model


def greedy_decode(tokenizers, transformer, sentence: tf.Tensor):
    assert isinstance(sentence, tf.Tensor)
    if tf.rank(sentence) == 0:
        sentence = sentence[tf.newaxis]

    # Токенизация вопроса
    enc = tokenizers.question.tokenize(sentence).to_tensor()

    start_id, end_id = tokenizers.answer.get_start_end_ids()
    start = start_id[tf.newaxis]
    end = end_id[tf.newaxis]

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(MAX_TOKENS):
        out_ids = tf.transpose(output_array.stack())  # [1, t]
        logits = transformer([enc, out_ids], training=False)  # [1, t, vocab]
        logits = logits[:, -1:, :]
        next_id = tf.argmax(logits, axis=-1)  # [1, 1]
        output_array = output_array.write(i + 1, next_id[0])

        if tf.reduce_all(tf.equal(next_id, end)):
            break

    out_ids = tf.transpose(output_array.stack())  # [1, T]
    text_out = tokenizers.answer.detokenize(out_ids)[0]  # scalar string
    return text_out, out_ids


def debug_tokenizers(tokenizers):
    print("=== DEBUG: tokenizers sanity check ===")
    s_q = tf.constant(["как дела?"])
    ids_q = tokenizers.question.tokenize(s_q)
    det_q = tokenizers.question.detokenize(ids_q)

    print("Q ids:", ids_q.to_list())
    print("Q det:", [x.decode("utf-8") for x in det_q.numpy()])

    s_a = tf.constant(["привет, всё ок"])
    ids_a = tokenizers.answer.tokenize(s_a)
    det_a = tokenizers.answer.detokenize(ids_a)

    print("A ids:", ids_a.to_list())
    print("A det:", [x.decode("utf-8") for x in det_a.numpy()])

    start_id, end_id = tokenizers.answer.get_start_end_ids()
    print("start_id:", start_id.numpy(), "end_id:", end_id.numpy())
    print("=" * 80)


def main():
    logger.info(
        "Загружаем HF-сплиты (для совместимости, но не используем их напрямую)..."
    )
    # Важно: create_or_load_model может ожидать, что HF уже загружен и vocab-ы существуют
    train_hf, val_hf = load_hf_splits()
    _train_examples, _val_examples = make_tf_datasets(train_hf, val_hf)

    logger.info("Грузим токенизаторы...")
    tokenizers = load_tokenizers()

    debug_tokenizers(tokenizers)

    logger.info("Создаём/грузим Transformer...")
    transformer = create_or_load_model(tokenizers)

    logger.info("Интерактивный debug greedy decode (без SavedModel).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        sentence = tf.constant(q)
        text_out, ids = greedy_decode(tokenizers, transformer, sentence)

        ids_np = ids.numpy()  # shape: (1, T)
        ids_list = ids_np.tolist()[0]

        print("\nRAW IDS:", ids_list)
        print("DETOK :", text_out.numpy().decode("utf-8"))
        print("-" * 80)

    logger.info("Выход.")


if __name__ == "__main__":
    main()
