from src.config import PROCESSED_DATA_DIR

# --- Directory settings ---
YAKU_DIR = PROCESSED_DATA_DIR / "yaku"

TRAIN_DIR = YAKU_DIR / "train"
VALID_DIR = YAKU_DIR / "valid"

INPUT_NAME = "input"
OUTPUT_NAME = "output"

# --- Dataset creation parameters ---
MAX_PLAYER_GAME_COUNT = 1000
TARGET_BATCH_SIZE = 100000
TRAIN_BATCH_COUNT = 13
VALID_BATCH_COUNT = 1

# --- Network architecture ---
INPUT_DIM = 472
HIDDEN_LAYERS = [512, 512, 256, 256, 256, 64]
OUTPUT_DIM = 1

# --- Training hyperparameters ---
LEARNING_RATE = 1e-3
LEARNING_BATCH_SIZE = 16384
