"""
predictor.py — Pipeline de traitement des radiographies pulmonaires

Ce module implémente les trois étapes du pipeline de diagnostic :
  1. Validation    : détection des images hors-distribution (OOD) avant toute inférence
  2. Prétraitement : normalisation de l'image vers un format 224×224 niveaux de gris
  3. Extraction HOG: calcul du descripteur Histogram of Oriented Gradients (576 features)
  4. Classification: prédiction via le modèle Pipeline(StandardScaler, SVM-RBF) entraîné

Les paramètres HOG correspondent exactement à ceux du notebook d'entraînement (rapport.ipynb)
afin de garantir la cohérence entre l'entraînement et l'inférence.

Fonctions supplémentaires :
  - generate_heatmap  : visualisation côte-à-côte de l'attention HOG
  - anonymize_image   : masquage des marges où les données patient sont gravées
"""

import os
import pickle
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

# ── Configuration du pipeline ─────────────────────────────────────────────────

MODEL_PATH    = os.getenv("MODEL_PATH", "model/pneumonia_hog_model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0-HOG-SVM")

IMG_SIZE = (224, 224)   # Taille cible (largeur × hauteur en pixels)

HOG_PARAMS = {
    "orientations":     4,          # Nombre de bins directionnels
    "pixels_per_cell":  (32, 32),   # Taille d'une cellule HOG
    "cells_per_block":  (2, 2),     # Nombre de cellules par bloc
    "block_norm":       "L2-Hys",   # Normalisation intra-bloc
}
# → Vecteur de features : nb_blocs_x × nb_blocs_y × cells_per_block² × orientations
#   = (224/32 - 1) × (224/32 - 1) × (2×2) × 4 = 7 × 7 × 4 × 4 = 576 dimensions

# ── Seuils de validation hors-distribution (OOD) ─────────────────────────────

# Une radiographie thoracique réelle est quasi-entièrement en niveaux de gris.
# Seuil de saturation : si trop de pixels sont colorés, l'image n'est pas une radio.
OOD_MAX_SATURATION_MEAN = 15.0      # Saturation HSV moyenne maximale autorisée

# Densité des contours (ratio pixels de bord / pixels totaux après Canny).
# Un document texte ou une photo naturelle présente une densité très élevée.
OOD_MAX_EDGE_DENSITY    = 0.35      # Densité maximale de contours autorisée

# ── Chargement du modèle (singleton) ─────────────────────────────────────────

_model = None  # Instance unique en mémoire pour éviter les rechargements répétés

def _load_model():
    """Charge le modèle depuis le disque lors du premier appel (lazy loading)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable : '{MODEL_PATH}'. "
                "Copiez pneumonia_hog_model.pkl dans le dossier backend/model/."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


# ── Étapes du pipeline ────────────────────────────────────────────────────────

def is_valid_xray(image_bytes: bytes) -> bool:
    """
    Garde hors-distribution (OOD) : vérifie qu'une image ressemble à une
    radiographie thoracique standard avant d'appliquer le modèle HOG+SVM.

    Problème résolu : HOG+SVM ne reconnaît pas les poumons — il détecte des
    gradients. Un document texte ou une photographie couleur peut déclencher un
    score de confiance de 97%+ de manière totalement artefactuelle. Cette fonction
    bloque ces cas avant l'inférence.

    Deux critères complémentaires sont utilisés :

    1. Saturation de couleur (espace HSV) :
       Les radiographies sont des images en niveaux de gris ; leur saturation HSV
       est quasi-nulle. Une image RGB naturelle (photo, document) présente une
       saturation nettement supérieure.

    2. Densité de contours (filtre de Canny) :
       Un cliché thoracique présente une densité modérée de contours (structures
       anatomiques). Un document texte ou un schéma technique présente une densité
       d'arêtes anormalement élevée.

    Returns:
        True si l'image satisfait les deux critères, False sinon.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False

    # Critère 1 : saturation HSV
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    if mean_saturation > OOD_MAX_SATURATION_MEAN:
        return False

    # Critère 2 : densité de contours Canny
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges        = cv2.Canny(gray, threshold1=50, threshold2=150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    if edge_density > OOD_MAX_EDGE_DENSITY:
        return False

    return True


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Étape 1 — Prétraitement de l'image.

    Réplique exactement la fonction preprocess_xray() du notebook d'entraînement :
      - Conversion en niveaux de gris
      - Redimensionnement proportionnel (thumbnail) vers 224×224 px
      - Centrage sur un canvas noir pour conserver le ratio d'aspect

    Returns:
        Tableau NumPy float32 de forme (224, 224).
    """
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img.thumbnail(IMG_SIZE, Image.BICUBIC)

    # Canvas noir 224×224 : évite la déformation sur les images non carrées
    canvas = Image.new("L", IMG_SIZE, color=0)
    offset_x = (IMG_SIZE[0] - img.size[0]) // 2
    offset_y = (IMG_SIZE[1] - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))

    return np.array(canvas, dtype=np.float32)


def _extract_hog(img: np.ndarray) -> np.ndarray:
    """
    Étape 2 — Extraction des features HOG.

    Returns:
        Vecteur NumPy de 576 features décrivant les gradients de l'image.
    """
    return hog(img, **HOG_PARAMS, visualize=False)


# ── Interface publique ────────────────────────────────────────────────────────

def predict_image(image_bytes: bytes) -> Tuple[str, float, str]:
    """
    Pipeline complet : octets d'image → prédiction clinique.

    Lève ValueError si l'image ne satisfait pas le garde OOD (is_valid_xray).
    Cette exception est capturée dans main.py et retournée comme HTTP 400.

    Returns:
        label       : 'PNEUMONIA' ou 'NORMAL'
        probability : score de confiance pour la classe PNEUMONIA [0.0 – 1.0]
        version     : identifiant de la version du modèle utilisé
    """
    if not is_valid_xray(image_bytes):
        raise ValueError(
            "Format d'image invalide : veuillez soumettre une radiographie thoracique standard (incidence PA). "
            "Les photographies, documents et images colorées ne sont pas acceptées."
        )

    model    = _load_model()
    img      = _preprocess(image_bytes)
    features = _extract_hog(img)

    label       = "PNEUMONIA" if model.predict([features])[0] == 1 else "NORMAL"
    probability = float(model.predict_proba([features])[0][1])  # P(PNEUMONIA)

    return label, probability, MODEL_VERSION


def generate_heatmap(image_bytes: bytes, output_path: str) -> None:
    """
    Génère une visualisation côte-à-côte (image originale | carte d'attention HOG).

    La carte d'attention est obtenue en récupérant l'image HOG brute (visualize=True),
    en la colorisant via la palette JET, puis en la superposant à l'image originale.
    """
    img  = _preprocess(image_bytes)
    gray = img.astype(np.uint8)

    # Calcul HOG avec visualisation activée
    _, hog_image = hog(img, **HOG_PARAMS, visualize=True)

    # Colorisation de la carte d'attention
    hog_norm    = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hog_colored = cv2.applyColorMap(hog_norm, cv2.COLORMAP_JET)
    gray_bgr    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay     = cv2.addWeighted(gray_bgr, 0.6, hog_colored, 0.4, 0)

    # Étiquettes
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray_bgr, "Original",      (8, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(overlay,  "HOG Attention", (8, 22), font, 0.6, (200, 200, 200), 1)

    # Assemblage côte-à-côte avec séparateur vertical
    separator = np.full((IMG_SIZE[1], 3, 3), 160, dtype=np.uint8)
    combined  = np.hstack([gray_bgr, separator, overlay])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, combined)


def anonymize_image(image_bytes: bytes) -> bytes:
    """
    Anonymise une radiographie en masquant les coins supérieurs et inférieurs
    par des rectangles noirs opaques.

    Justification : les systèmes PACS grèvent les données démographiques du
    patient (nom, date de naissance, identifiant hospitalier) dans les coins
    de l'image. Un masquage par rectangle noir est plus robuste qu'un floutage :
    il est irréversible et élimine tout risque de reconstruction par déconvolution.

    Les zones masquées couvrent :
      - Coins supérieurs gauche et droit  : 6% × 25% de l'image
      - Coins inférieurs gauche et droit  : 6% × 25% de l'image

    Returns:
        Octets de l'image anonymisée (PNG), ou image originale si le décodage échoue.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return image_bytes  # Image illisible : retourner sans modification

    h, w = img.shape

    # Hauteur et largeur des zones de masquage
    mask_h = int(h * 0.06)   # 6% de la hauteur (zone de texte PACS)
    mask_w = int(w * 0.25)   # 25% de la largeur pour chaque coin

    # Application des rectangles noirs dans les quatre coins
    img[:mask_h, :mask_w]    = 0   # Coin supérieur gauche
    img[:mask_h, -mask_w:]   = 0   # Coin supérieur droit
    img[-mask_h:, :mask_w]   = 0   # Coin inférieur gauche
    img[-mask_h:, -mask_w:]  = 0   # Coin inférieur droit

    success, encoded = cv2.imencode(".png", img)
    return encoded.tobytes() if success else image_bytes
