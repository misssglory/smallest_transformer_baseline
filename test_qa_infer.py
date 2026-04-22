# test_qa_infer.py

import tensorflow as tf
import tensorflow_text as text  # noqa: F401  # регистрирует TF Text ops
from loguru import logger

from config import EXPORTED_TRANSLATOR_DIR, LOG_LEVEL


def setup_logger():
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}\n",
        level=LOG_LEVEL,
    )


def load_inference_module():
    logger.info(f"Загружаем InferenceModule из {EXPORTED_TRANSLATOR_DIR}")
    return tf.saved_model.load(str(EXPORTED_TRANSLATOR_DIR))


def main():
    setup_logger()
    infer = load_inference_module()

    # Печатаем доступные сигнатуры
    print("Available signatures:", list(getattr(infer, "signatures", {}).keys()))
    if hasattr(infer, "signatures") and "serving_default" in infer.signatures:
        fn = infer.signatures["serving_default"]
        print("serving_default input_signature:", fn.structured_input_signature)
        print("serving_default outputs:", fn.structured_outputs)

    logger.info("Интерактивный режим. Введите вопрос (или пустую строку для выхода).")

    while True:
        try:
            q = input("> ").strip()
            logger.info(q)
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        x = tf.constant(q)

        # 1) Прямой вызов __call__
        res = infer(x)
        print("\nDIRECT CALL RESULT type:", type(res))

        if isinstance(res, dict):
            text_t = res.get("text", None)
            ids_t = res.get("ids", None)
        else:
            text_t = res
            ids_t = None

        if text_t is not None:
            raw = text_t.numpy()
            print("TEXT tensor:", raw)
            if isinstance(raw, bytes):
                print("TEXT decoded:", raw.decode("utf-8"))

        if ids_t is not None:
            ids_np = ids_t.numpy()
            print("IDS shape:", ids_np.shape)
            print("IDS:", ids_np.tolist())

        # 2) Вызов через signature (на всякий случай)
        if hasattr(infer, "signatures") and "serving_default" in infer.signatures:
            fn = infer.signatures["serving_default"]
            sig_res = fn(sentence=x)
            print("\nSIGNATURE RESULT type:", type(sig_res))
            print("SIGNATURE keys:", list(sig_res.keys()))

            text_t2 = sig_res.get("text", None)
            ids_t2 = sig_res.get("ids", None)

            if text_t2 is not None:
                raw2 = text_t2.numpy()
                print("SIG TEXT tensor:", raw2)
                if isinstance(raw2, bytes):
                    print("SIG TEXT decoded:", raw2.decode("utf-8"))

            if ids_t2 is not None:
                ids_np2 = ids_t2.numpy()
                print("SIG IDS shape:", ids_np2.shape)
                print("SIG IDS:", ids_np2.tolist())

        print("-" * 80)

    logger.info("Выход.")


if __name__ == "__main__":
    main()
