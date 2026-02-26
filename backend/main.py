"""
main.py — Point d'entrée de l'API REST PneumoScan AI

Organisation des endpoints :
  POST /auth/token              Authentification, retourne un token JWT
  GET  /auth/me                 Informations sur l'utilisateur courant

  POST /sessions                Ouvre une session de travail
  PUT  /sessions/{id}/close     Clôture une session
  GET  /sessions                Liste toutes les sessions

  POST /predict                 Soumet une radiographie pour analyse
  GET  /history                 Historique des prédictions (filtrable)
  GET  /heatmap/{filename}      Sert l'image de carte d'attention HOG

  GET  /dashboard               KPIs et statistiques (admin uniquement)
  GET  /health                  Vérification de l'état du serveur
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm

from auth import (Token, User, authenticate_user, create_access_token,
                  require_admin, require_radiologist)
from database import (close_session, compute_triage, create_session,
                      get_dashboard_stats, get_predictions, get_session,
                      get_sessions, init_db, log_prediction)
from predictor import anonymize_image, generate_heatmap, predict_image
from utils import validate_image

# ── Application et configuration ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "PneumoScan AI",
    description = "API d'aide au diagnostic de pneumonie par analyse HOG + SVM",
    version     = "2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HEATMAP_DIR = os.getenv("HEATMAP_DIR", "/app/heatmaps")


@app.on_event("startup")
def startup() -> None:
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    init_db()
    logger.info("PneumoScan AI démarré — base de données initialisée.")


# ── Authentification ──────────────────────────────────────────────────────────

@app.post("/auth/token", response_model=Token, tags=["Auth"],
          summary="Connexion utilisateur — retourne un token JWT")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Identifiants incorrects.")
    token = create_access_token(username=user["username"], role=user["role"])
    return Token(access_token=token, token_type="bearer",
                 role=user["role"], full_name=user["full_name"])


@app.get("/auth/me", tags=["Auth"], summary="Informations sur l'utilisateur courant")
def me(user: User = Depends(require_radiologist)):
    return user


# ── Sessions ──────────────────────────────────────────────────────────────────

@app.post("/sessions", tags=["Sessions"], summary="Ouvrir une session de travail")
def open_session(
    patient_count: int  = Query(..., ge=1, le=500, description="Nombre de patients à analyser"),
    user:          User = Depends(require_radiologist),
):
    session = create_session(patient_count=patient_count, operator=user.username)
    logger.info(f"Session ouverte : id={session['id']} | {patient_count} patients | opérateur={user.username}")
    return session


@app.put("/sessions/{session_id}/close", tags=["Sessions"], summary="Clôturer une session")
def end_session(session_id: str, user: User = Depends(require_radiologist)):
    if not get_session(session_id):
        raise HTTPException(status_code=404, detail="Session introuvable.")
    result = close_session(session_id)
    logger.info(f"Session clôturée : id={session_id}")
    return result


@app.get("/sessions", tags=["Sessions"], summary="Lister toutes les sessions")
def list_sessions(user: User = Depends(require_radiologist)):
    return get_sessions()


# ── Analyse ───────────────────────────────────────────────────────────────────

@app.post("/predict", tags=["Diagnostic"],
          summary="Analyser une radiographie thoracique")
async def predict(
    file:       UploadFile      = File(..., description="Radiographie JPG ou PNG"),
    session_id: Optional[str]   = Query(None, description="Identifiant de la session active"),
    anonymize:  bool            = Query(False, description="Appliquer l'anonymisation des marges"),
    user:       User            = Depends(require_radiologist),
):
    # 1. Validation du fichier entrant
    if not validate_image(file):
        raise HTTPException(status_code=400, detail="Format non supporté. Utilisez JPG ou PNG (max 10 Mo).")

    image_bytes = await file.read()

    # 2. Anonymisation optionnelle (floutage des marges contenant les données patient)
    if anonymize:
        image_bytes = anonymize_image(image_bytes)

    # 3. Inférence : validation OOD → prétraitement → HOG → SVM
    try:
        label, probability, model_version = predict_image(image_bytes)
    except ValueError as e:
        # Image hors-distribution (pas une radiographie) : 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        # Modèle non trouvé : 503 Service Unavailable
        raise HTTPException(status_code=503, detail=str(e))

    triage = compute_triage(prediction=label, probability=probability)

    # 4. Génération de la carte d'attention HOG
    heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
    heatmap_path     = os.path.join(HEATMAP_DIR, heatmap_filename)
    generate_heatmap(image_bytes, heatmap_path)

    # 5. Persistance en base de données
    meta = log_prediction(
        image_name    = file.filename,
        prediction    = label,
        probability   = probability,
        model_version = model_version,
        heatmap_path  = heatmap_path,
        session_id    = session_id,
        operator      = user.username,
        anonymized    = anonymize,
    )

    logger.info(
        f"Analyse : id={meta['id']} | fichier={file.filename} | "
        f"résultat={label} ({probability:.1%}) | triage={triage} | session={session_id}"
    )

    return {
        "id":            meta["id"],
        "timestamp":     meta["timestamp"],
        "prediction":    label,
        "probability":   round(probability, 4),
        "triage_level":  triage,
        "model_version": model_version,
        "heatmap_url":   f"/heatmap/{heatmap_filename}",
        "operator":      user.username,
        "anonymized":    anonymize,
    }


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
    rows = get_predictions(date_from, date_to, session_id, triage_level, limit)
    return {"count": len(rows), "predictions": rows}


# ── Tableau de bord BI ────────────────────────────────────────────────────────

@app.get("/dashboard", tags=["Admin"],
         summary="KPIs et statistiques — réservé à l'administrateur")
def dashboard(user: User = Depends(require_admin)):
    return get_dashboard_stats()


# ── Ressources statiques ──────────────────────────────────────────────────────

@app.get("/heatmap/{filename}", tags=["Diagnostic"],
         summary="Servir une carte d'attention HOG générée")
def serve_heatmap(filename: str):
    path = os.path.join(HEATMAP_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Carte d'attention introuvable.")
    return FileResponse(path, media_type="image/png")


@app.get("/health", tags=["System"], summary="Vérification de l'état du serveur")
def health():
    return {"status": "ok", "version": "2.0.0", "timestamp": datetime.utcnow().isoformat()}
