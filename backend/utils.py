"""Validation helpers for uploaded images.

This module centralizes upload validation so all ingestion paths enforce the
same security and quality constraints.
"""

import logging
from typing import Optional

from fastapi import UploadFile

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/x-png",
    "image/dicom",
    "application/dicom",
    "application/x-dicom",
    "application/dicom+json",
    "application/octet-stream",
}
ALLOWED_EXTENSIONS    = {"jpg", "jpeg", "png", "dcm", "dicom"}
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _extract_extension(filename: Optional[str]) -> Optional[str]:
    """Safely extract a lowercase extension without leading dot."""
    if not filename or "." not in filename:
        return None
    return filename.rsplit(".", 1)[-1].lower()


def validate_upload_metadata(filename: Optional[str], content_type: Optional[str]) -> bool:
    """Validate filename extension and client-declared MIME metadata.

    Browsers and PACS exports are inconsistent for medical files: some send
    empty MIME types, others send generic ``application/octet-stream`` or
    non-standard DICOM values. We keep a strict extension check and apply a
    tolerant MIME policy so valid radiographies are not incorrectly rejected.
    """
    extension = _extract_extension(filename)
    if extension is None or extension not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected file: unsupported extension in filename '%s'", filename)
        return False

    normalized = (content_type or "").split(";", 1)[0].strip().lower()

    # Missing/unknown MIME from browser: trust extension validation.
    if not normalized:
        return True

    if normalized in ALLOWED_CONTENT_TYPES:
        return True

    # Accept common image/* variants when extension is an image extension.
    if normalized.startswith("image/") and extension in {"jpg", "jpeg", "png"}:
        return True

    # Accept vendor-specific DICOM MIME values as long as extension is DICOM.
    if "dicom" in normalized and extension in {"dcm", "dicom"}:
        return True

    logger.warning("Rejected file: unsupported MIME type '%s' for filename '%s'", content_type, filename)
    return False


def validate_image_bytes(payload: bytes) -> bool:
    """Validate upload payload size and non-empty content."""
    size = len(payload)
    return 0 < size <= MAX_UPLOAD_SIZE_BYTES


def validate_image(file: UploadFile) -> bool:
    """Backward-compatible metadata validation for FastAPI UploadFile."""
    return validate_upload_metadata(filename=file.filename, content_type=file.content_type)
