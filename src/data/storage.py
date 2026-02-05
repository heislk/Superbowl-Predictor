import json
import os
import logging
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

def get_file_path(filename: str, processed: bool = False) -> str:
    directory = PROCESSED_DATA_DIR if processed else RAW_DATA_DIR
    return os.path.join(directory, filename)

def save_json(filename: str, data: any, processed: bool = False) -> None:
    filepath = get_file_path(filename, processed)
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")

def load_json(filename: str, processed: bool = False) -> any:
    filepath = get_file_path(filename, processed)
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        return None

def file_exists(filename: str, processed: bool = False) -> bool:
    return os.path.exists(get_file_path(filename, processed))
