from pathlib import Path

REQUESTS_TIMEOUT = 30

_CONFIG_FILE_PATH = Path(__file__).resolve()
SRC_ROOT = _CONFIG_FILE_PATH.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
LOG_ZIPS_DIR = RAW_DATA_DIR / "log_zips"

TENHO_RAW_DATA_URL = "https://tenhou.net/sc/raw/"
TENHO_LOG_ZIP_FILENAME_FORMAT = "scraw{year}.zip"
TENHO_LOG_ZIP_URL_FORMAT = f"{TENHO_RAW_DATA_URL}{TENHO_LOG_ZIP_FILENAME_FORMAT}"
TENHO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Referer": TENHO_RAW_DATA_URL,
}
