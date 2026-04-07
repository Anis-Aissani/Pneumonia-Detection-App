"""Batch scanner for incoming chest X-ray files.

The scanner watches an input directory and submits each new file to the API.
Successfully processed files are moved to a dedicated folder.
"""

import os
import time
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCAN_DIR = os.getenv("SCAN_DIR", "../incoming_scans")
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))
PROCESSED_DIR = os.path.join(SCAN_DIR, "processed")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}


@dataclass(frozen=True)
class ScannerConfig:
    """Runtime settings for the scanner worker."""

    scan_dir: str = SCAN_DIR
    processed_dir: str = PROCESSED_DIR
    api_url: str = API_URL
    poll_interval: int = POLL_INTERVAL


def process_file(filepath: str, config: ScannerConfig) -> None:
    """Submit one file to the prediction endpoint and move it on success."""
    filename = os.path.basename(filepath)
    logger.info("[SCANNER] Processing: %s", filename)

    with open(filepath, "rb") as f:
        ext = os.path.splitext(filename)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".png":
            mime = "image/png"
        else:
            mime = "application/dicom"
        response = requests.post(
            f"{config.api_url}/predict",
            files={"file": (filename, f, mime)},
            timeout=30,
        )

    if response.status_code == 200:
        result = response.json()
        logger.info(
            "[SCANNER] Success: %s -> %s (%.1f%%) | id=%s",
            filename,
            result["prediction"],
            result["probability"] * 100,
            result["id"],
        )
        os.makedirs(config.processed_dir, exist_ok=True)
        os.replace(filepath, os.path.join(config.processed_dir, filename))
    else:
        logger.error("[SCANNER] Failed: %s -> HTTP %s: %s", filename, response.status_code, response.text)


def _iter_pending_files(scan_dir: str) -> list[str]:
    """Return processable image files in deterministic order."""
    filepaths: list[str] = []
    for fname in sorted(os.listdir(scan_dir)):
        path = os.path.join(scan_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in ALLOWED_EXTENSIONS and os.path.isfile(path):
            filepaths.append(path)
    return filepaths


def scan_loop(config: ScannerConfig | None = None) -> None:
    """Continuously process files from the watch directory."""
    cfg = config or ScannerConfig()
    os.makedirs(cfg.scan_dir, exist_ok=True)
    logger.info("[SCANNER] Watching %s every %ss", cfg.scan_dir, cfg.poll_interval)

    while True:
        try:
            for path in _iter_pending_files(cfg.scan_dir):
                try:
                    process_file(path, cfg)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("[SCANNER] Error processing %s: %s", os.path.basename(path), exc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("[SCANNER] Loop error: %s", exc)

        time.sleep(cfg.poll_interval)


if __name__ == "__main__":
    scan_loop()
