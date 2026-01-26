from src.yaku.common.config import YAKU_DATA_DIR, YAKU_RESULT_DIR

# --- WandB settings ---
GROUP_NAME = "exp4"

# --- Directory settings ---
DATA_DIR = YAKU_DATA_DIR / "exp4"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

RESULT_DIR = YAKU_RESULT_DIR / "exp4"

DATA_STATS_CSV = RESULT_DIR / "data_stats.csv"

INPUT_FILENAME_PREFIX = "input"
OUTPUT_FILENAME_PREFIX = "output"

# --- Dataset creation parameters ---
CHUNK_SIZE = 100000

# --- Network architecture ---
SEQ_LEN = 35
TOKEN_FEATURE_DIM = 16
D_MODEL = 128
NUM_HEADS = 8
NUM_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
OUTPUT_DIM = 33

# --- Training hyperparameters ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LEARNING_BATCH_SIZE = 2048
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 5
