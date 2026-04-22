# train_qa.py

import tensorflow as tf
import tensorflow_text as text  # noqa: F401  # регистрирует ops токенизатора
from loguru import logger
from tqdm.keras import TqdmCallback

from data_pipeline import load_hf_splits, make_tf_datasets
from tokenization import create_tokenizers, load_tokenizers
from model import create_or_load_model, prepare_batch_fn, make_batches
from config import (
    LOG_LEVEL,
    EPOCHS,
    EXPORTED_TRANSLATOR_DIR,
    MAX_TOKENS,
    TRAINED_MODEL_PATH,
    TRUNCATE_DATASET_FOR_DEBUG,
)


class InferenceModule(tf.Module):
    def __init__(self, tokenizers, transformer):
        super().__init__()
        self.tokenizers = tokenizers
        self.transformer = transformer

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        if tf.rank(sentence) == 0:
            sentence = sentence[tf.newaxis]

        enc = self.tokenizers.question.tokenize(sentence).to_tensor()

        start_id, end_id = self.tokenizers.answer.get_start_end_ids()
        start = start_id[tf.newaxis]
        end = end_id[tf.newaxis]

        ta = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        ta = ta.write(0, start)

        for i in tf.range(MAX_TOKENS):
            out_ids = tf.transpose(ta.stack())  # [1, t]
            logits = self.transformer([enc, out_ids], training=False)
            logits = logits[:, -1:, :]
            next_id = tf.argmax(logits, axis=-1)  # [1, 1]
            ta = ta.write(i + 1, next_id[0])

            if tf.reduce_all(tf.equal(next_id, end)):
                break

        out_ids = tf.transpose(ta.stack())  # [1, T]
        text_out = self.tokenizers.answer.detokenize(out_ids)[0]

        return {
            "text": text_out,
            "ids": out_ids,
        }


def greedy_decode(tokenizers, transformer, sentence: tf.Tensor):
    if tf.rank(sentence) == 0:
        sentence = sentence[tf.newaxis]

    enc = tokenizers.question.tokenize(sentence).to_tensor()

    start_id, end_id = tokenizers.answer.get_start_end_ids()
    start = start_id[tf.newaxis]
    end = end_id[tf.newaxis]

    ta = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    ta = ta.write(0, start)

    for i in tf.range(MAX_TOKENS):
        out_ids = tf.transpose(ta.stack())
        logits = transformer([enc, out_ids], training=False)
        logits = logits[:, -1:, :]
        next_id = tf.argmax(logits, axis=-1)
        ta = ta.write(i + 1, next_id[0])

        if tf.reduce_all(tf.equal(next_id, end)):
            break

    out_ids = tf.transpose(ta.stack())
    text_out = tokenizers.answer.detokenize(out_ids)[0]
    return text_out, out_ids


def main():
    best_weights_path = TRAINED_MODEL_PATH.with_name(
        TRAINED_MODEL_PATH.name.replace(".weights.h5", ".best.weights.h5")
    )
    logger.info(f"Best weights path: {best_weights_path}")

    # 1) HF-сплиты и tf.data (без repeat)
    train_hf, val_hf = load_hf_splits()
    train_examples, val_examples = make_tf_datasets(train_hf, val_hf)

    # 2) Токенизаторы
    try:
        tokenizers = load_tokenizers()
        logger.info("Токенизаторы загружены.")
    except Exception:
        logger.info("Токенизаторы не найдены, создаём...")
        tokenizers = create_tokenizers(train_examples)

    # 3) Батчи
    prepare_batch = prepare_batch_fn(tokenizers)
    train_batches = make_batches(train_examples, prepare_batch)
    val_batches = make_batches(val_examples, prepare_batch)

    # 4) Модель (+ возможная загрузка весов внутри create_or_load_model)
    transformer = create_or_load_model(tokenizers)

    # 5) Колбэки
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_weights_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    tqdm_callback = TqdmCallback(verbose=1)

    # 6) Обучение с tqdm + EarlyStopping + best checkpoint
    logger.info("Запускаем обучение/дообучение...")

    if TRUNCATE_DATASET_FOR_DEBUG > 0:
        train_batches = train_batches.take(TRUNCATE_DATASET_FOR_DEBUG)

    logger.info(f"Train batches size: {len(list(train_batches))}")

    history = transformer.fit(
        train_batches.take(3333),
        epochs=EPOCHS,
        validation_data=val_batches,
        callbacks=[
            tqdm_callback,
            early_stopping,
            model_checkpoint,
        ],
        verbose=0,
    )

    logger.info(
        f"Train loss: {history.history['loss'][-1]:.4f}, "
        f"Train acc: {history.history['masked_accuracy'][-1]:.4f}, "
        f"Val loss: {history.history['val_loss'][-1]:.4f}, "
        f"Val acc: {history.history['val_masked_accuracy'][-1]:.4f}"
    )

    # 7) Явно загружаем лучшие веса перед экспортом, если checkpoint появился
    if best_weights_path.exists():
        logger.info(f"Загружаем лучшие веса из {best_weights_path}")
        transformer.load_weights(str(best_weights_path))

    # 8) Sanity-декод сразу после обучения / загрузки лучших весов
    try:
        test_q = tf.constant("как дела?")
        text_out, ids = greedy_decode(tokenizers, transformer, test_q)
        logger.info(
            f"Sanity decode: ids={ids.numpy().tolist()} "
            f"text={text_out.numpy().decode('utf-8')!r}"
        )
    except Exception as e:
        logger.exception(f"Sanity decode failed: {e}")

    # 9) Сохраняем финальные веса модели
    logger.info(f"Сохраняем финальные веса модели в {TRAINED_MODEL_PATH}")
    transformer.save_weights(str(TRAINED_MODEL_PATH))

    # 10) Экспорт InferenceModule
    infer_module = InferenceModule(tokenizers, transformer)
    logger.info(f"Сохраняем InferenceModule в {EXPORTED_TRANSLATOR_DIR}")
    tf.saved_model.save(infer_module, export_dir=str(EXPORTED_TRANSLATOR_DIR))

    logger.info("Готово.")


if __name__ == "__main__":
    main()
