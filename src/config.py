from pathlib import Path

# ---- General Settings ----
DB_BATCH_SIZE = 1000
DB_CHUNK_SIZE = 500000
REQUEST_SLEEP_MIN = 0.5
REQUEST_SLEEP_MAX = 1.5
SUBPROCESS_TIMEOUT = 30

# ---- Path Settings ----
_CONFIG_FILEPATH = Path(__file__).resolve()
SRC_ROOT = _CONFIG_FILEPATH.parent
PROJECT_ROOT = SRC_ROOT.parent

MODEL_DIR = PROJECT_ROOT / "models"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
DUMPS_DIR = DATA_DIR / "dumps"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

LOG_ZIPS_DIR = RAW_DATA_DIR / "log_zips"
GZIPPED_LOGS_DIR = INTERIM_DATA_DIR / "gzipped_logs"
TEXT_LOGS_DIR = INTERIM_DATA_DIR / "text_logs"
MJLOGS_DIR = INTERIM_DATA_DIR / "mjlogs"
JSON_LOGS_DIR = INTERIM_DATA_DIR / "json_logs"
