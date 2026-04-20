# test_qa_infer.py

import tensorflow as tf
import tensorflow_text as text
from loguru import logger

from config import EXPORTED_TRANSLATOR_DIR, LOG_LEVEL


def load_exported_translator():
    logger.info(f"Загружаем ExportTranslator из {EXPORTED_TRANSLATOR_DIR}")
    return tf.saved_model.load(str(EXPORTED_TRANSLATOR_DIR))


def main():

    translator = load_exported_translator()

    logger.info("Интерактивный режим. Введите вопрос (или пустую строку для выхода).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        out = translator(tf.constant(q))
        print(out.numpy().decode("utf-8"))

    logger.info("Выход.")


if __name__ == "__main__":
    main()
