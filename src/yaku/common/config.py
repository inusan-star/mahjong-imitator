from src.config import PROCESSED_DATA_DIR, RESULTS_DIR

# --- WandB settings ---
PROJECT_NAME = "yaku"

# --- Directory settings ---
YAKU_DATA_DIR = PROCESSED_DATA_DIR / "yaku"
YAKU_RESULT_DIR = RESULTS_DIR / "yaku"
SPLITS_FILE = YAKU_DATA_DIR / "splits.json"
YAKU_DISTRIBUTION_FILE = YAKU_RESULT_DIR / "yaku_distribution.csv"

# --- Dataset split parameters ---
SEED = 42
TOTAL_EXTRACT_ROUNDS = 1000000
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

# --- Research Constraints ---
YAKU_THRESHOLD_RATIO = 0.0001
EXCLUDED_YAKU_NAMES = ["一発", "槍槓", "嶺上開花", "海底摸月", "河底撈魚", "両立直", "天和", "地和", "裏ドラ"]
