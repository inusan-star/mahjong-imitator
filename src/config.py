from pathlib import Path

# ---- General Settings ----
DB_BATCH_SIZE = 1000
REQUESTS_TIMEOUT = 30
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
PROCESSED_DATA_DIR = DATA_DIR / "processed"

LOG_ZIPS_DIR = RAW_DATA_DIR / "log_zips"
GZIPPED_LOGS_DIR = INTERIM_DATA_DIR / "gzipped_logs"
TEXT_LOGS_DIR = INTERIM_DATA_DIR / "text_logs"
MJLOGS_DIR = INTERIM_DATA_DIR / "mjlogs"
JSON_LOGS_DIR = INTERIM_DATA_DIR / "json_logs"

# ---- Tenhou Crawler Settings ----
TENHO_RAW_DATA_URL = "https://tenhou.net/sc/raw/"
TENHO_LOG_ZIP_FILENAME_FORMAT = "scraw{year}.zip"
TENHO_LOG_ZIP_URL_FORMAT = f"{TENHO_RAW_DATA_URL}{TENHO_LOG_ZIP_FILENAME_FORMAT}"
TENHO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": TENHO_RAW_DATA_URL,
}
TENHO_LOG_TIME_REGEX = r"^\d{2}:\d{2}$"
TENHO_LOG_ID_REGEX = r"log=([0-9]{10}gm-[0-9a-f-]+)"
TENHO_LOG_URL_FORMAT = "http://tenhou.net/0/log/?{source_id}"

# ---- MJLOG Parsing Settings ----
MJLOG_GO_REGEX = r'<GO\s+type="(\d+)"\s+lobby="(\d+)"[^>]*/>'
MJLOG_UN_REGEX = r'<UN\s+n0="([^"]+)"\s+n1="([^"]+)"\s+n2="([^"]+)"\s+n3="([^"]+)"\s+dan="([^"]+)"\s+rate="([^"]+)"\s+sx="([^"]+)"[^>]*/>'
MJLOG_OWARI_REGEX = r'owari="(.*?)"'
