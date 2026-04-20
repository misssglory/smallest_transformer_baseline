# train_qa.py

import tensorflow as tf
from loguru import logger

from data_pipeline import load_hf_splits, make_tf_datasets
from tokenization import create_tokenizers, load_tokenizers
from model import create_or_load_model, prepare_batch_fn, make_batches

from config import (
    LOG_LEVEL,
    EPOCHS,
    TRAINED_MODEL_PATH,
    EXPORTED_TRANSLATOR_DIR,
    MAX_TOKENS,
)


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

        start_id, end_id = self.tokenizers.answer.get_start_end_ids()
        start = start_id[tf.newaxis]
        end = end_id[tf.newaxis]

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


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens, attention_weights = self.translator(
            sentence, max_length=MAX_TOKENS
        )
        return result


def main():
    train_hf, val_hf = load_hf_splits()
    train_examples, val_examples = make_tf_datasets(train_hf, val_hf)

    try:
        tokenizers = load_tokenizers()
    except Exception:
        tokenizers = create_tokenizers(train_examples)

    prepare_batch = prepare_batch_fn(tokenizers)
    train_batches = make_batches(train_examples, prepare_batch)
    val_batches = make_batches(val_examples, prepare_batch)
    transformer = create_or_load_model(tokenizers)

    logger.info("Запускаем обучение/дообучение...")
    transformer.fit(train_batches[:200], epochs=EPOCHS, validation_data=val_batches)

    translator = Translator(tokenizers, transformer)
    export_translator = ExportTranslator(translator)

    logger.info(f"Сохраняем экспортируемый переводчик в {EXPORTED_TRANSLATOR_DIR}")
    tf.saved_model.save(export_translator, export_dir=str(EXPORTED_TRANSLATOR_DIR))

    logger.info("Готово.")


if __name__ == "__main__":
    main()
