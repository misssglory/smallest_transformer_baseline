# smoke_save_test.py

import tensorflow as tf
import tensorflow_text as text
from loguru import logger

from config import EXPORTED_TRANSLATOR_DIR, MAX_TOKENS
from data_pipeline import load_hf_splits, make_tf_datasets
from tokenization import create_tokenizers
from model import Transformer, CustomSchedule, masked_loss, masked_accuracy, prepare_batch_fn, make_batches


def main():
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n",
        level="INFO",
    )

    # --- 1. маленький датасет ---
    train_hf, val_hf = load_hf_splits()
    train_hf_small = train_hf.select(range(min(500, len(train_hf))))
    val_hf_small = val_hf.select(range(min(100, len(val_hf))))
    train_examples, val_examples = make_tf_datasets(train_hf_small, val_hf_small)

    # --- 2. токенизаторы (vocab уже есть, так что быстро) ---
    tokenizers = create_tokenizers(train_examples)

    input_vocab_size = int(tokenizers.question.get_vocab_size().numpy())
    target_vocab_size = int(tokenizers.answer.get_vocab_size().numpy())

    # --- 3. маленькая модель ---
    transformer = Transformer(
        num_layers=1,
        d_model=64,
        num_heads=4,
        dff=128,
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        dropout_rate=0.1,
    )

    lr = CustomSchedule(64)
    optimizer = tf.keras.optimizers.Adam(
        lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )
    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy],
    )

    prepare_batch = prepare_batch_fn(tokenizers)
    train_batches = make_batches(train_examples, prepare_batch)
    val_batches = make_batches(val_examples, prepare_batch)

    logger.info("Короткое обучение для smoke-теста (несколько батчей, 1 эпоха)...")
    transformer.fit(train_batches.take(10), epochs=1, validation_data=val_batches.take(2))

    # --- 4. InferenceModule с @tf.function ---
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

            output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
            output_array = output_array.write(0, start)

            for i in tf.range(MAX_TOKENS):
                out_ids = tf.transpose(output_array.stack())
                logits = self.transformer([enc, out_ids], training=False)
                logits = logits[:, -1:, :]
                next_id = tf.argmax(logits, axis=-1)  # [1,1]
                output_array = output_array.write(i + 1, next_id[0])
                if tf.reduce_all(tf.equal(next_id, end)):
                    break

            out_ids = tf.transpose(output_array.stack())
            text_out = self.tokenizers.answer.detokenize(out_ids)[0]
            return text_out

    infer_module = InferenceModule(tokenizers, transformer)

    logger.info(f"Сохраняем InferenceModule в {EXPORTED_TRANSLATOR_DIR}")
    tf.saved_model.save(infer_module, export_dir=str(EXPORTED_TRANSLATOR_DIR))

    # --- 5. Перезагрузка и быстрый тест ---
    logger.info("Перезагружаем InferenceModule и проверяем инференс...")
    reloaded = tf.saved_model.load(str(EXPORTED_TRANSLATOR_DIR))

    sample_q = "как дела?"
    out = reloaded(tf.constant(sample_q))
    print(f"Q: {sample_q}")
    print("A:", out.numpy().decode("utf-8"))


if __name__ == "__main__":
    main()
