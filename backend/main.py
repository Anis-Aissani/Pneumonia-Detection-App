"""FastAPI entrypoint for PneumoScan backend.

The module now follows a controller-service style:
- Controllers (routes) handle HTTP concerns.
- Services orchestrate business workflows.
- Data and ML modules provide infrastructure operations.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm

from auth import (Token, User, authenticate_user, create_access_token,
                  require_admin, require_radiologist)
from database import init_db
from services import DashboardService, HistoryService, PredictionService, SessionService
from settings import settings

# ── Application et configuration ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = PredictionService(heatmap_dir=settings.heatmap_dir)
session_service = SessionService()
history_service = HistoryService()
dashboard_service = DashboardService()


@app.on_event("startup")
def startup() -> None:
    os.makedirs(settings.heatmap_dir, exist_ok=True)
    init_db()
    logger.info("PneumoScan API started and database initialized.")


# ── Authentification ──────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=Token, tags=["Auth"],
          summary="Connexion utilisateur — retourne un token JWT")
def login(form: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user credentials and return a JWT access token."""
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    token = create_access_token(username=user["username"], role=user["role"])
    return Token(access_token=token, token_type="bearer",
                 role=user["role"], full_name=user["full_name"])


@app.get("/auth/me", tags=["Auth"], summary="Informations sur l'utilisateur courant")
def me(user: User = Depends(require_radiologist)):
    """Return current authenticated user profile."""
    return user


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.post("/sessions", tags=["Sessions"], summary="Ouvrir une session de travail")
def open_session(
    patient_count: int  = Query(..., ge=1, le=500, description="Nombre de patients à analyser"),
    user:          User = Depends(require_radiologist),
):
    """Open a clinical reading session."""
    session = session_service.open_session(patient_count=patient_count, operator=user.username)
    logger.info(f"Session ouverte : id={session['id']} | {patient_count} patients | opérateur={user.username}")
    return session


@app.put("/sessions/{session_id}/close", tags=["Sessions"], summary="Clôturer une session")
def end_session(session_id: str, user: User = Depends(require_radiologist)):
    """Close an existing work session."""
    try:
        result = session_service.close_session(session_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    logger.info(f"Session clôturée : id={session_id}")
    return result


@app.get("/sessions", tags=["Sessions"], summary="Lister toutes les sessions")
def list_sessions(user: User = Depends(require_radiologist)):
    """List all work sessions with activity counts."""
    return session_service.list_sessions()


# ── Analyse ───────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["Diagnostic"],
          summary="Analyser une radiographie thoracique")
async def predict(
    file:       UploadFile      = File(..., description="Radiographie JPG, PNG ou DICOM (.dcm)"),
    session_id: Optional[str]   = Query(None, description="Identifiant de la session active"),
    anonymize:  bool            = Query(False, description="Appliquer l'anonymisation des marges"),
    user:       User            = Depends(require_radiologist),
):
    """Run a chest X-ray inference request and return triage-ready output."""
    image_bytes = await file.read()
    try:
        result = prediction_service.run_prediction(
            filename=file.filename or "upload",
            content_type=file.content_type,
            image_bytes=image_bytes,
            operator=user.username,
            session_id=session_id,
            should_anonymize=anonymize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        logger.info(
            "Analyse: id=%s file=%s result=%s (%.1f%%) triage=%s session=%s",
            result["id"],
            file.filename,
            result["prediction"],
            result["probability"] * 100,
            result["triage_level"],
            session_id,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# ── Historique ────────────────────────────────────────────────────────────────

@app.get("/history", tags=["Diagnostic"],
         summary="Historique des analyses (filtrable par date, session, triage)")
def history(
    date_from:    Optional[str] = Query(None, description="Date de début (YYYY-MM-DD)"),
    date_to:      Optional[str] = Query(None, description="Date de fin (YYYY-MM-DD)"),
    session_id:   Optional[str] = Query(None),
    triage_level: Optional[str] = Query(None, description="CRITICAL | MODERATE | ROUTINE"),
    limit:        int           = Query(500, ge=1, le=1000),
    user:         User          = Depends(require_radiologist),
):
    """Return prediction history with optional filtering."""
    rows = history_service.list_predictions(date_from, date_to, session_id, triage_level, limit)
    return {"count": len(rows), "predictions": rows}


# ── Tableau de bord BI ────────────────────────────────────────────────────────

@app.get("/dashboard", tags=["Admin"],
         summary="KPIs et statistiques — réservé à l'administrateur")
def dashboard(user: User = Depends(require_admin)):
    """Return BI dashboard metrics for administrators."""
    return dashboard_service.get_dashboard()


# ── Ressources statiques ──────────────────────────────────────────────────────

@app.get("/heatmap/{filename}", tags=["Diagnostic"],
         summary="Servir une carte d'attention HOG générée")
def serve_heatmap(filename: str):
    """Serve generated heatmap images by filename."""
    path = os.path.join(settings.heatmap_dir, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Carte d'attention introuvable.")
    return FileResponse(path, media_type="image/png")


@app.get("/health", tags=["System"], summary="Vérification de l'état du serveur")
def health():
    """Liveness endpoint used by orchestration and monitoring."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
