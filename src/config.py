from pathlib import Path

REQUESTS_TIMEOUT = 30

_CONFIG_FILEPATH = Path(__file__).resolve()
SRC_ROOT = _CONFIG_FILEPATH.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"

LOG_ZIPS_DIR = RAW_DATA_DIR / "log_zips"
GZIPPED_LOGS_DIR = INTERIM_DATA_DIR / "gzipped_logs"
TEXT_LOGS_DIR = INTERIM_DATA_DIR / "text_logs"

TENHO_RAW_DATA_URL = "https://tenhou.net/sc/raw/"
TENHO_LOG_ZIP_FILENAME_FORMAT = "scraw{year}.zip"
TENHO_LOG_ZIP_URL_FORMAT = f"{TENHO_RAW_DATA_URL}{TENHO_LOG_ZIP_FILENAME_FORMAT}"
TENHO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": TENHO_RAW_DATA_URL,
}
TENHO_LOG_ID_REGEX = r"log=([0-9]{10}gm-[0-9a-f-]+)"
TENHO_LOG_URL_FORMAT = "http://tenhou.net/0/log/?{log_id}"
