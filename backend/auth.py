"""
auth.py — Authentification JWT avec contrôle d'accès basé sur les rôles (RBAC)

Deux rôles sont définis :
  - 'radiologist' : peut créer des sessions et soumettre des analyses
  - 'admin'       : accès complet, incluant le tableau de bord BI

Architecture : les utilisateurs sont stockés en mémoire pour ce prototype.
En production, ils seraient stockés en base de données avec leurs mots de passe hachés.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Configuration ─────────────────────────────────────────────────────────────

SECRET_KEY = os.getenv("SECRET_KEY", "pneumoscan-dev-key-change-in-production")
ALGORITHM  = "HS256"
TOKEN_TTL_MINUTES = 480  # Durée de vie du token = 1 shift hospitalier (8h)

pwd_context  = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Base de données utilisateurs (en mémoire) ─────────────────────────────────

USERS: dict = {
    "radiologist": {
        "username":        "radiologist",
        "full_name":       "Dr. Martin Dubois",
        "role":            "radiologist",
        "hashed_password": pwd_context.hash("radio123"),
    },
    "admin": {
        "username":        "admin",
        "full_name":       "Administrateur Système",
        "role":            "admin",
        "hashed_password": pwd_context.hash("admin123"),
    },
}

# ── Modèles Pydantic ──────────────────────────────────────────────────────────

class User(BaseModel):
    username:  str
    full_name: str
    role:      str

class Token(BaseModel):
    access_token: str
    token_type:   str
    role:         str
    full_name:    str

# ── Fonctions d'authentification ──────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Vérifie les identifiants et retourne l'utilisateur, ou None si invalide."""
    user = USERS.get(username)
    if user and pwd_context.verify(password, user["hashed_password"]):
        return user
    return None


def create_access_token(username: str, role: str) -> str:
    """Crée un token JWT signé contenant l'identité et le rôle de l'utilisateur."""
    payload = {
        "sub":  username,
        "role": role,
        "exp":  datetime.utcnow() + timedelta(minutes=TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dépendance FastAPI : décode le token JWT et retourne l'utilisateur courant."""
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role     = payload.get("role")
        if not username:
            raise ValueError
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expirée ou token invalide. Veuillez vous reconnecter.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = USERS.get(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Utilisateur introuvable.")
    return User(username=username, full_name=user["full_name"], role=role)


def require_role(*allowed_roles: str):
    """
    Fabrique de dépendances : génère un vérificateur de rôle pour un endpoint.

    Exemple d'usage :
        @app.get("/admin-only")
        def route(user = Depends(require_role("admin"))):
            ...
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Accès refusé. Rôle requis : {', '.join(allowed_roles)}.",
            )
        return user
    return _check


# Alias lisibles pour les dépendances d'endpoints
require_radiologist = require_role("radiologist", "admin")
require_admin       = require_role("admin")
