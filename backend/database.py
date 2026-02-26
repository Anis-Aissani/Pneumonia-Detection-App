"""
database.py — Couche d'accès aux données (SQLite)

Gère deux entités principales :
  - Session    : groupe de patients analysés lors d'un même shift de travail
  - Prediction : résultat d'une analyse radiographique, lié à une session

Triage automatique : chaque prédiction reçoit un niveau de priorité calculé
à partir du score de confiance du modèle et du résultat clinique.
"""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH = os.getenv("DB_PATH", "/app/data/predictions.db")

# Seuils de triage clinique (inspirés des protocoles de tri hospitalier)
TRIAGE_CRITICAL_THRESHOLD = 0.85   # Probabilité > 85% → révision immédiate
TRIAGE_ROUTINE_THRESHOLD  = 0.15   # Probabilité < 15% → file standard

# ── Utilitaires de connexion ──────────────────────────────────────────────────

@contextmanager
def get_db():
    """
    Gestionnaire de contexte pour les connexions SQLite.
    Garantit la fermeture de la connexion même en cas d'exception.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Accès aux colonnes par nom
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Crée les tables et index si inexistants. Idempotent."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id            TEXT PRIMARY KEY,
                timestamp     TEXT NOT NULL,
                image_name    TEXT NOT NULL,
                prediction    TEXT NOT NULL,          -- 'PNEUMONIA' ou 'NORMAL'
                probability   REAL NOT NULL,          -- Score de confiance [0.0 – 1.0]
                model_version TEXT NOT NULL,
                heatmap_path  TEXT,
                triage_level  TEXT NOT NULL,          -- 'CRITICAL', 'MODERATE', 'ROUTINE'
                session_id    TEXT,
                operator      TEXT,
                anonymized    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id            TEXT PRIMARY KEY,
                started_at    TEXT NOT NULL,
                closed_at     TEXT,
                patient_count INTEGER NOT NULL,
                operator      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'open'  -- 'open' ou 'closed'
            );

            -- Index pour les filtres fréquents
            CREATE INDEX IF NOT EXISTS idx_pred_session   ON predictions(session_id);
            CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pred_triage    ON predictions(triage_level);
        """)


# ── Logique métier : triage ───────────────────────────────────────────────────

def compute_triage(prediction: str, probability: float) -> str:
    """
    Calcule le niveau de triage selon la règle clinique définie :
      - CRITICAL : pneumonie détectée avec forte confiance (> 85%)
      - MODERATE : pneumonie détectée avec confiance intermédiaire
      - ROUTINE  : normal, ou pneumonie avec faible confiance (< 15%)
    """
    if prediction == "PNEUMONIA" and probability > TRIAGE_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if prediction == "PNEUMONIA":
        return "MODERATE"
    return "ROUTINE"


# ── CRUD Predictions ──────────────────────────────────────────────────────────

def log_prediction(
    image_name:    str,
    prediction:    str,
    probability:   float,
    model_version: str,
    heatmap_path:  Optional[str] = None,
    session_id:    Optional[str] = None,
    operator:      Optional[str] = None,
    anonymized:    bool = False,
) -> Dict:
    """Enregistre une prédiction et retourne ses métadonnées."""
    pred_id      = str(uuid.uuid4())
    timestamp    = datetime.utcnow().isoformat()
    triage_level = compute_triage(prediction, probability)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, timestamp, image_name, prediction, probability, model_version,
                heatmap_path, triage_level, session_id, operator, anonymized)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (pred_id, timestamp, image_name, prediction, round(probability, 4),
             model_version, heatmap_path, triage_level, session_id, operator,
             int(anonymized)),
        )

    return {"id": pred_id, "timestamp": timestamp, "triage_level": triage_level}


def get_predictions(
    date_from:    Optional[str] = None,
    date_to:      Optional[str] = None,
    session_id:   Optional[str] = None,
    triage_level: Optional[str] = None,
    limit:        int = 500,
) -> List[Dict]:
    """
    Récupère les prédictions avec filtres optionnels.
    Les résultats sont triés par priorité de triage (CRITICAL en premier),
    puis par date décroissante.
    """
    query  = "SELECT * FROM predictions WHERE 1=1"
    params: list = []

    if date_from:
        query += " AND timestamp >= ?";    params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?";    params.append(date_to + "T23:59:59")
    if session_id:
        query += " AND session_id = ?";    params.append(session_id)
    if triage_level:
        query += " AND triage_level = ?";  params.append(triage_level)

    # Tri prioritaire : CRITICAL → MODERATE → ROUTINE, puis chronologique inversé
    query += """
        ORDER BY CASE triage_level
            WHEN 'CRITICAL' THEN 1
            WHEN 'MODERATE' THEN 2
            ELSE 3
        END, timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ── CRUD Sessions ─────────────────────────────────────────────────────────────

def close_stale_sessions(operator: str) -> int:
    """
    Clôture toutes les sessions ouvertes appartenant à cet opérateur.

    Appelée automatiquement à chaque connexion pour éliminer les sessions
    « fantômes » laissées ouvertes si l'utilisateur a fermé son navigateur
    sans cliquer sur Déconnexion (le JWT était stateless, la session DB ne l'est pas).

    Returns:
        Nombre de sessions fermées.
    """
    closed_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET status='closed', closed_at=? WHERE operator=? AND status='open'",
            (closed_at, operator),
        )
        return cursor.rowcount


def create_session(patient_count: int, operator: str) -> Dict:
    """
    Ouvre une nouvelle session de travail.
    Les éventuelles sessions fantômes précédentes sont clôturées avant création.
    """
    close_stale_sessions(operator)  # Failsafe anti-zombie sessions

    session = {
        "id":            str(uuid.uuid4()),
        "started_at":    datetime.utcnow().isoformat(),
        "patient_count": patient_count,
        "operator":      operator,
        "status":        "open",
    }
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, started_at, patient_count, operator) VALUES (?, ?, ?, ?)",
            (session["id"], session["started_at"], patient_count, operator),
        )
    return session


def close_session(session_id: str) -> Dict:
    """Clôture une session et retourne ses données mises à jour."""
    closed_at = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE sessions SET status='closed', closed_at=? WHERE id=?",
            (closed_at, session_id),
        )
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else {}


def get_session(session_id: str) -> Optional[Dict]:
    """Retourne une session par son identifiant."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def get_sessions(limit: int = 50) -> List[Dict]:
    """Retourne toutes les sessions avec le nombre d'analyses effectuées."""
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.*, COUNT(p.id) AS scans_done
            FROM sessions s
            LEFT JOIN predictions p ON p.session_id = s.id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── Tableau de bord BI ────────────────────────────────────────────────────────

def get_dashboard_stats() -> Dict:
    """
    Calcule les indicateurs clés de performance (KPIs) pour le tableau de bord.
    Toutes les agrégations sont réalisées en une seule passe sur la base de données.
    """
    with get_db() as conn:

        # KPIs globaux en une seule requête agrégée
        kpis = conn.execute("""
            SELECT
                COUNT(*)                                          AS total_scans,
                SUM(prediction = 'PNEUMONIA')                    AS pneumonia_count,
                SUM(triage_level = 'CRITICAL')                   AS critical_count,
                (SELECT COUNT(*) FROM sessions)                  AS sessions_total
            FROM predictions
        """).fetchone()

        # Volume quotidien sur les 14 derniers jours
        daily = conn.execute("""
            SELECT
                substr(timestamp, 1, 10)              AS day,
                COUNT(*)                              AS total,
                SUM(prediction = 'PNEUMONIA')         AS pneumonia_count
            FROM   predictions
            WHERE  timestamp >= datetime('now', '-14 days')
            GROUP  BY day
            ORDER  BY day
        """).fetchall()

        # Répartition par niveau de triage
        triage_dist = conn.execute("""
            SELECT triage_level, COUNT(*) AS count
            FROM   predictions
            GROUP  BY triage_level
        """).fetchall()

    total    = kpis["total_scans"] or 0
    pneumonia = kpis["pneumonia_count"] or 0

    return {
        "total_scans":            total,
        "pneumonia_count":        pneumonia,
        "normal_count":           total - pneumonia,
        "critical_count":         kpis["critical_count"] or 0,
        "sessions_total":         kpis["sessions_total"] or 0,
        "positivity_rate":        round(pneumonia / total * 100, 1) if total else 0.0,
        # Indicateur ROI : estimation basée sur 5 min de lecture manuelle par radio
        "estimated_hours_saved":  round(total * 5 / 60, 1),
        "daily_stats":            [dict(r) for r in daily],
        "triage_distribution":    [dict(r) for r in triage_dist],
    }
