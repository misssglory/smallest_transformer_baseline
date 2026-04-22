# config.py

from pathlib import Path

# Корневая папка проекта
PROJECT_ROOT = Path(__file__).resolve().parent

# Пути
QUESTION_VOCAB_PATH = PROJECT_ROOT / "question_vocab.txt"
ANSWER_VOCAB_PATH = PROJECT_ROOT / "answer_vocab.txt"

TOKENIZERS_SAVEDMODEL_DIR = PROJECT_ROOT / "russian_dialogues_qa_converter"
# TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_transformer"
TRAINED_MODEL_PATH = PROJECT_ROOT / "trained_transformer.weights.h5"

EXPORTED_TRANSLATOR_DIR = PROJECT_ROOT / "translator"
# EXPORTED_TRANSLATOR_DIR = PROJECT_ROOT / "russian_dialogues_qa_converter"

# Датасет
HF_DATASET_NAME = "Den4ikAI/russian_dialogues"
HF_SPLIT = "train"
HF_TEST_SIZE = 0.01
HF_SEED = 42

# Токенизация
VOCAB_SIZE = 8000
MAX_TOKENS = 40
RESERVED_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]
BERT_TOKENIZER_PARAMS = dict(lower_case=True)

# Обучение
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
EPOCHS = 1
BUFFER_SIZE = 20000
BATCH_SIZE = 64
TRUNCATE_DATASET_FOR_DEBUG=-1

# Логирование
LOG_LEVEL = "INFO"
