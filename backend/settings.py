"""Application configuration for PneumoScan backend.

Centralizing environment-based settings keeps entrypoints thin and makes
configuration easier to explain during a project defense.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    """Immutable runtime settings loaded from environment variables."""

    app_name: str = "PneumoScan"
    app_version: str = "2.1.0"
    app_description: str = "API d'aide au triage de pneumonie (EfficientNet+SVM prioritaire, HOG+SVM fallback)"
    heatmap_dir: str = os.getenv("HEATMAP_DIR", "/app/heatmaps")
    cors_origins: tuple[str, ...] = ("*",)


settings = AppSettings()
