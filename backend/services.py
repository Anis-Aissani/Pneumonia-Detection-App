"""Application services used by FastAPI routes.

These services isolate orchestration logic from HTTP concerns. This improves
modularity and keeps endpoints focused on request/response concerns.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Optional

from database import (
    close_session,
    compute_triage,
    create_session,
    get_dashboard_stats,
    get_predictions,
    get_session,
    get_sessions,
    log_prediction,
)
from predictor import anonymize_image, generate_heatmap, predict_image
from utils import validate_image_bytes, validate_upload_metadata


@dataclass
class PredictionService:
    """Coordinates the full prediction workflow from file to persisted result."""

    heatmap_dir: str

    def __post_init__(self) -> None:
        os.makedirs(self.heatmap_dir, exist_ok=True)

    def run_prediction(
        self,
        *,
        filename: str,
        content_type: Optional[str],
        image_bytes: bytes,
        operator: str,
        session_id: Optional[str],
        should_anonymize: bool,
    ) -> dict:
        """Run validation, optional anonymization, inference, heatmap and persistence."""
        if not validate_upload_metadata(filename=filename, content_type=content_type):
            raise ValueError("Format non supporte. Utilisez JPG, PNG ou DICOM (.dcm) (max 10 Mo).")
        if not validate_image_bytes(image_bytes):
            raise ValueError("Le fichier image est vide ou depasse la limite autorisee (10 Mo).")

        input_bytes = anonymize_image(image_bytes) if should_anonymize else image_bytes
        label, probability, model_version = predict_image(input_bytes)
        triage_level = compute_triage(prediction=label, probability=probability)

        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join(self.heatmap_dir, heatmap_filename)
        generate_heatmap(input_bytes, heatmap_path)

        meta = log_prediction(
            image_name=filename,
            prediction=label,
            probability=probability,
            model_version=model_version,
            heatmap_path=heatmap_path,
            session_id=session_id,
            operator=operator,
            anonymized=should_anonymize,
        )

        return {
            "id": meta["id"],
            "timestamp": meta["timestamp"],
            "prediction": label,
            "probability": round(probability, 4),
            "triage_level": triage_level,
            "model_version": model_version,
            "heatmap_url": f"/heatmap/{heatmap_filename}",
            "operator": operator,
            "anonymized": should_anonymize,
        }


class SessionService:
    """Session-related use cases."""

    @staticmethod
    def open_session(patient_count: int, operator: str) -> dict:
        return create_session(patient_count=patient_count, operator=operator)

    @staticmethod
    def close_session(session_id: str) -> dict:
        if not get_session(session_id):
            raise LookupError("Session introuvable.")
        return close_session(session_id)

    @staticmethod
    def list_sessions() -> list[dict]:
        return get_sessions()


class HistoryService:
    """History and triage list retrieval use cases."""

    @staticmethod
    def list_predictions(
        date_from: Optional[str],
        date_to: Optional[str],
        session_id: Optional[str],
        triage_level: Optional[str],
        limit: int,
    ) -> list[dict]:
        return get_predictions(date_from, date_to, session_id, triage_level, limit)


class DashboardService:
    """Administrative BI dashboard use cases."""

    @staticmethod
    def get_dashboard() -> dict:
        return get_dashboard_stats()
