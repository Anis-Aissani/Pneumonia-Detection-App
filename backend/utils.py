"""
utils.py — Validation des fichiers entrants

Centralise les règles d'acceptation des radiographies soumises à l'API.
Deux vérifications sont effectuées en défense en profondeur :
  1. Type MIME déclaré par le client HTTP (facilement falsifiable)
  2. Extension du nom de fichier (vérification indépendante côté serveur)
"""

import logging
from fastapi import UploadFile

logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS    = {"jpg", "jpeg", "png"}


def validate_image(file: UploadFile) -> bool:
    """
    Valide le type MIME et l'extension d'un fichier uploadé.

    Returns:
        True si le fichier est un JPEG ou PNG valide, False sinon.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Fichier rejeté — type MIME non supporté : {file.content_type}")
        return False

    if file.filename:
        extension = file.filename.lower().rsplit(".", 1)[-1]
        if extension not in ALLOWED_EXTENSIONS:
            logger.warning(f"Fichier rejeté — extension non supportée : {extension}")
            return False

    return True
