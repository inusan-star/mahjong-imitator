from src.yaku.common.config import YAKU_DATA_DIR, YAKU_RESULT_DIR

# --- WandB settings ---
GROUP_NAME = "exp2"

# --- Directory settings ---
DATA_DIR = YAKU_DATA_DIR / "exp1"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

RESULT_DIR = YAKU_RESULT_DIR / "exp2"

DATA_STATS_CSV = RESULT_DIR / "data_stats.csv"

INPUT_FILENAME_PREFIX = "input"
OUTPUT_FILENAME_PREFIX = "output"

# --- Dataset creation parameters ---
CHUNK_SIZE = 100000

# --- Network architecture ---
INPUT_DIM = 472
OUTPUT_DIM = 33
HIDDEN_LAYERS = [512, 512, 256, 256, 256, 64]

# --- Training hyperparameters ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LEARNING_BATCH_SIZE = 2048
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
