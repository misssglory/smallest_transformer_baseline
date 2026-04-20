# test_qa_infer.py

import pprint
from typing import Any

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


def describe_loaded_model(infer: Any):
    logger.info("=== DEBUG: loaded object info ===")

    attrs = [a for a in dir(infer) if not a.startswith("_")]
    print("Public attrs:")
    pprint.pprint(attrs)

    if hasattr(infer, "signatures"):
        print("\nAvailable signatures:")
        pprint.pprint(list(infer.signatures.keys()))
        for name, fn in infer.signatures.items():
            print(f"\nSignature: {name}")
            print("  structured_input_signature =", fn.structured_input_signature)
            print("  structured_outputs        =", fn.structured_outputs)
    else:
        print("\nNo .signatures found on loaded object.")

    print("\nCallable:", callable(infer))
    print("=" * 80)


def debug_tensor(name: str, value: tf.Tensor):
    print(f"[{name}]")
    print("  type: ", type(value))
    print("  dtype:", getattr(value, "dtype", None))
    print("  shape:", getattr(value, "shape", None))

    try:
        raw = value.numpy()
        print("  numpy:", raw)

        if isinstance(raw, bytes):
            try:
                print("  decoded:", raw.decode("utf-8"))
            except Exception as e:
                print("  decoded: <decode error>", repr(e))
        elif hasattr(raw, "tolist"):
            print("  tolist:", raw.tolist())
    except Exception as e:
        print("  numpy() failed:", repr(e))

    print("-" * 80)


def debug_output(result: Any):
    print("\n=== DEBUG: inference result ===")
    print("Python type:", type(result))

    if isinstance(result, tf.Tensor):
        debug_tensor("result", result)
        return

    if isinstance(result, dict):
        print("Result is dict with keys:", list(result.keys()))
        for k, v in result.items():
            if isinstance(v, tf.Tensor):
                debug_tensor(f"dict[{k!r}]", v)
            else:
                print(f"[dict[{k!r}]] type={type(v)} value={v}")
                print("-" * 80)
        return

    if isinstance(result, (tuple, list)):
        print(f"Result is {type(result).__name__} of len={len(result)}")
        for i, v in enumerate(result):
            if isinstance(v, tf.Tensor):
                debug_tensor(f"result[{i}]", v)
            else:
                print(f"[result[{i}]] type={type(v)} value={v}")
                print("-" * 80)
        return

    print("Raw result:", result)
    print("=" * 80)


def extract_display_text(result: Any) -> str:
    """
    Пытаемся аккуратно извлечь текст для печати пользователю.
    Поддерживаем:
      - scalar tf.string tensor
      - dict с первым tf.string tensor
      - tuple/list, где есть tf.string tensor
    """
    if isinstance(result, tf.Tensor):
        raw = result.numpy()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        if hasattr(raw, "tolist"):
            return str(raw.tolist())
        return str(raw)

    if isinstance(result, dict):
        for _, v in result.items():
            if isinstance(v, tf.Tensor):
                raw = v.numpy()
                if isinstance(raw, bytes):
                    return raw.decode("utf-8")
                if hasattr(raw, "tolist"):
                    return str(raw.tolist())
        return str(result)

    if isinstance(result, (tuple, list)):
        for v in result:
            if isinstance(v, tf.Tensor):
                raw = v.numpy()
                if isinstance(raw, bytes):
                    return raw.decode("utf-8")
                if hasattr(raw, "tolist"):
                    return str(raw.tolist())
        return str(result)

    return str(result)


def try_call_direct(infer, q: str):
    logger.info("DEBUG: direct call infer(tf.constant(q))")
    result = infer(tf.constant(q))
    debug_output(result)
    text_out = extract_display_text(result)
    print(f"DISPLAY TEXT: {text_out!r}")
    return result


def try_call_signature(infer, q: str):
    if not hasattr(infer, "signatures") or "serving_default" not in infer.signatures:
        logger.info("DEBUG: serving_default signature not found, skip.")
        return None

    fn = infer.signatures["serving_default"]
    logger.info("DEBUG: call infer.signatures['serving_default'](...)")

    try:
        result = fn(tf.constant(q))
    except TypeError:
        # Иногда signature требует keyword arg
        _, kw = fn.structured_input_signature
        if len(kw) == 1:
            key = next(iter(kw.keys()))
            result = fn(**{key: tf.constant(q)})
        else:
            raise

    debug_output(result)
    text_out = extract_display_text(result)
    print(f"DISPLAY TEXT (signature): {text_out!r}")
    return result


def main():
    setup_logger()
    infer = load_inference_module()
    describe_loaded_model(infer)

    logger.info("Интерактивный debug-режим. Введите вопрос (или пустую строку для выхода).")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            break

        print("\n" + "=" * 80)
        print(f"INPUT: {q!r}")
        print("=" * 80)

        try:
            direct_result = try_call_direct(infer, q)
        except Exception as e:
            logger.exception(f"Direct call failed: {e}")
            direct_result = None

        try:
            signature_result = try_call_signature(infer, q)
        except Exception as e:
            logger.exception(f"Signature call failed: {e}")
            signature_result = None

        print("\nSUMMARY")
        print("  direct_result_type   =", type(direct_result))
        print("  signature_result_type=", type(signature_result))
        print("-" * 80)

    logger.info("Выход.")


if __name__ == "__main__":
    main()
