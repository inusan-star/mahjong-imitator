from src.config import PROCESSED_DATA_DIR

# --- WandB settings ---
PROJECT_NAME = "yaku"
GROUP_NAME = "exp1"

# --- Directory settings ---
YAKU_DIR = PROCESSED_DATA_DIR / "yaku" / "exp1"

TRAIN_DIR = YAKU_DIR / "train"
VALID_DIR = YAKU_DIR / "valid"
TEST_DIR = YAKU_DIR / "test"

INPUT_NAME = "input"
OUTPUT_NAME = "output"

# --- Dataset creation parameters ---
SEED = 42
START_YEAR = 2024
TARGET_BATCH_SIZE = 100000
TRAIN_BATCH_COUNT = 12
VALID_BATCH_COUNT = 1
TEST_BATCH_COUNT = 1

# --- Network architecture ---
INPUT_DIM = 472
HIDDEN_LAYERS = [512, 512, 256, 256, 256, 64]
OUTPUT_DIM = 1

# --- Training hyperparameters ---
LEARNING_RATE = 2e-4
LEARNING_BATCH_SIZE = 2048
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
