"""
scanner.py - Simulates a hospital scan workflow.
Watches /incoming_scans for new X-ray images and auto-processes them.

Run standalone: python scanner.py
Or as a background thread from main.py if needed.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCAN_DIR = os.getenv("SCAN_DIR", "../incoming_scans")
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))  # seconds
PROCESSED_DIR = os.path.join(SCAN_DIR, "processed")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def process_file(filepath: str):
    filename = os.path.basename(filepath)
    logger.info(f"[SCANNER] Processing: {filename}")

    with open(filepath, "rb") as f:
        ext = os.path.splitext(filename)[1].lower()
        mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (filename, f, mime)},
            timeout=30,
        )

    if response.status_code == 200:
        result = response.json()
        logger.info(
            f"[SCANNER] ✓ {filename} → {result['prediction']} "
            f"({result['probability']*100:.1f}%) | id={result['id']}"
        )
        # Move to processed/
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.rename(filepath, os.path.join(PROCESSED_DIR, filename))
    else:
        logger.error(f"[SCANNER] ✗ {filename} → HTTP {response.status_code}: {response.text}")


def scan_loop():
    os.makedirs(SCAN_DIR, exist_ok=True)
    logger.info(f"[SCANNER] Watching {SCAN_DIR} every {POLL_INTERVAL}s...")

    seen = set()
    while True:
        try:
            for fname in os.listdir(SCAN_DIR):
                fpath = os.path.join(SCAN_DIR, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext in ALLOWED_EXTENSIONS and fpath not in seen and os.path.isfile(fpath):
                    seen.add(fpath)
                    try:
                        process_file(fpath)
                    except Exception as e:
                        logger.error(f"[SCANNER] Error processing {fname}: {e}")
        except Exception as e:
            logger.error(f"[SCANNER] Loop error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    scan_loop()
