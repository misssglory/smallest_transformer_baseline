# train_qa.py

import tensorflow as tf
import tensorflow_text as text  # важно: регистрирует ops токенизатора
from loguru import logger

from data_pipeline import load_hf_splits, make_tf_datasets
from tokenization import create_tokenizers, load_tokenizers
from model import create_or_load_model, prepare_batch_fn, make_batches
from config import (
    LOG_LEVEL,
    EPOCHS,
    EXPORTED_TRANSLATOR_DIR,
    MAX_TOKENS,
)


class InferenceModule(tf.Module):
    def __init__(self, tokenizers, transformer):
        super().__init__()
        self.tokenizers = tokenizers
        self.transformer = transformer

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        # sentence: scalar string → [1] batch
        if tf.rank(sentence) == 0:
            sentence = sentence[tf.newaxis]

        # encoder input
        enc = self.tokenizers.question.tokenize(sentence).to_tensor()

        # start/end tokens
        start_id, end_id = self.tokenizers.answer.get_start_end_ids()
        start = start_id[tf.newaxis]
        end = end_id[tf.newaxis]

        # greedy decode
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(MAX_TOKENS):
            out_ids = tf.transpose(output_array.stack())  # [1, t]
            logits = self.transformer([enc, out_ids], training=False)
            logits = logits[:, -1:, :]  # [1, 1, vocab]
            next_id = tf.argmax(logits, axis=-1)  # [1, 1]
            output_array = output_array.write(i + 1, next_id[0])  # писать в TA

            if tf.reduce_all(tf.equal(next_id, end)):
                break

        out_ids = tf.transpose(output_array.stack())  # [1, T]
        text_out = self.tokenizers.answer.detokenize(out_ids)[0]  # scalar string
        return text_out


def main():
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n",
        level=LOG_LEVEL,
    )

    # 1) Загружаем HF-сплиты и tf.data
    train_hf, val_hf = load_hf_splits()
    train_examples, val_examples = make_tf_datasets(train_hf, val_hf)

    # 2) Токенизаторы
    try:
        tokenizers = load_tokenizers()
    except Exception:
        tokenizers = create_tokenizers(train_examples)

    # 3) Батчи
    prepare_batch = prepare_batch_fn(tokenizers)
    train_batches = make_batches(train_examples, prepare_batch)
    val_batches = make_batches(val_examples, prepare_batch)

    # 4) Модель
    transformer = create_or_load_model(tokenizers)

    # 5) Обучение (если хочешь ограничить датасет — используй .take(N))
    logger.info("Запускаем обучение/дообучение...")
    transformer.fit(train_batches.take(200), epochs=EPOCHS, validation_data=val_batches)

    # 6) Экспорт InferenceModule как SavedModel
    infer_module = InferenceModule(tokenizers, transformer)

    logger.info(f"Сохраняем InferenceModule в {EXPORTED_TRANSLATOR_DIR}")
    tf.saved_model.save(infer_module, export_dir=str(EXPORTED_TRANSLATOR_DIR))

    logger.info("Готово.")


if __name__ == "__main__":
    main()
