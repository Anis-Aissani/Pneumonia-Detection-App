"""
predictor.py — Pipeline de traitement des radiographies pulmonaires

Le backend est compatible avec deux variantes de modèles :

1) HOG + PCA + SVM (déployée par défaut)
2) EfficientNet-B0 + SVM (mode aligné résultats binôme)

Pourquoi cette double compatibilité :
- garder une exécution immédiate avec le modèle déjà présent dans le repo,
- permettre le passage à la variante EfficientNet-SVM dès que l'artefact est fourni.
"""

import os
import pickle
from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH    = os.getenv("MODEL_PATH", "model/pneumonia_effnet_svm.pkl")
MODEL_VERSION = "1.1.0-BINOME-HOG-PCA130-SVM"

IMG_SIZE = (224, 224)

HOG_PARAMS = {
    "orientations":    4,           # bins directionnels
    "pixels_per_cell": (32, 32),    # taille d'une cellule
    "cells_per_block": (2, 2),      # cellules par bloc
    "block_norm":      "L2-Hys",    # normalisation intra-bloc
}
# → (224/32 - 1)² × (2×2) × 4 = 7×7×4×4 = 576 features

# Seuils de validation OOD (Out-Of-Distribution)
OOD_MAX_SATURATION = 15.0   # radiographies ≈ niveaux de gris
OOD_MAX_EDGE_DENSITY = 0.22  # documents / interfaces ont une densité de contours plus élevée
OOD_MAX_STRAIGHT_LINE_DENSITY = 22.0  # lignes detectees par megapixel
OOD_MAX_BRIGHT_PIXEL_RATIO = 0.35     # grands aplats blancs (documents/UI) sont atypiques
OOD_MIN_GRAY_STD = 18.0               # reject images trop plates (peu d'information radiographique)

# ── Chargement du modèle (singleton) ─────────────────────────────────────────

_pkg = None
_effnet = None


def _is_dicom_bytes(image_bytes: bytes) -> bool:
    """Heuristic to detect DICOM payloads (DICM preamble or .dcm-like binary)."""
    return len(image_bytes) > 132 and image_bytes[128:132] == b"DICM"


def _decode_dicom_grayscale(image_bytes: bytes) -> np.ndarray:
    """Decode DICOM bytes to a uint8 grayscale image suitable for CV/ML processing."""
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    except Exception as exc:
        raise RuntimeError("Lecture DICOM indisponible: installez pydicom.") from exc

    ds = pydicom.dcmread(BytesIO(image_bytes), force=True)
    if not hasattr(ds, "PixelData"):
        raise ValueError("Fichier DICOM sans donnees pixel exploitables.")

    arr = ds.pixel_array
    if arr.ndim > 2:
        arr = arr[..., 0]

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)

    return (arr * 255.0).astype(np.uint8)


def _load_efficientnet_extractor() -> Optional[object]:
    """Charge EfficientNet-B0 en lazy loading si PyTorch est disponible."""
    global _effnet
    if _effnet is not None:
        return _effnet

    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except Exception:
        return None

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor.to(device)

    _effnet = {
        "torch": torch,
        "extractor": extractor,
        "device": device,
        "weights": models.EfficientNet_B0_Weights.DEFAULT,
    }
    return _effnet


def _load_model() -> dict:
    """Charge le modèle depuis le disque lors du premier appel (lazy loading)."""
    global _pkg
    if _pkg is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable : '{MODEL_PATH}'. "
                "Copiez pneumonia_hog_model.pkl dans backend/model/."
            )
        with open(MODEL_PATH, "rb") as f:
            _pkg = pickle.load(f)
    return _pkg


# ── Pipeline de traitement ────────────────────────────────────────────────────

def is_valid_xray(image_bytes: bytes) -> bool:
    """
    Garde hors-distribution : vérifie qu'une image ressemble à une radiographie.

    Critère 1 — Saturation HSV :
        Une radiographie est quasi-entièrement en niveaux de gris (saturation ≈ 0).
        Toute image colorée (photo, document) est rejetée (seuil : 15).

    Critère 2 — Densité de contours Canny :
        Un document texte ou un schéma a une densité d'arêtes anormalement élevée.
        Une radiographie présente une densité modérée.

    Critère 3 — Densité de lignes droites :
        Les captures d'écran/interfaces contiennent beaucoup de lignes droites
        horizontales/verticales (fenêtres, barres, tableaux).

    Critère 4 — Ratio de blancs saturés :
        Les documents et UIs ont de grands aplats très clairs, rares en RX thorax.

    Critère 5 — Contraste minimal :
        Une image trop uniforme (capture quasi-plate, image vide) est rejetée.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        try:
            gray = _decode_dicom_grayscale(image_bytes)
        except Exception:
            return False
        saturation = 0.0
    else:
        saturation = float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].mean())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if saturation > OOD_MAX_SATURATION:
        return False

    gray_std = float(gray.std())
    if gray_std < OOD_MIN_GRAY_STD:
        return False

    bright_ratio = float(np.mean(gray > 245))
    if bright_ratio > OOD_MAX_BRIGHT_PIXEL_RATIO:
        return False

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    if edge_density > OOD_MAX_EDGE_DENSITY:
        return False

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(30, int(min(gray.shape[:2]) * 0.12)),
        maxLineGap=8,
    )
    line_count = 0 if lines is None else len(lines)
    mpix = max(gray.shape[0] * gray.shape[1] / 1_000_000.0, 1e-6)
    if (line_count / mpix) > OOD_MAX_STRAIGHT_LINE_DENSITY:
        return False

    return True


def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Prétraitement identique au notebook du binôme :
      - Conversion en niveaux de gris
      - Redimensionnement proportionnel (thumbnail) → 224×224 px
      - Centrage sur un canvas noir (conserve le ratio d'aspect)

    Returns : tableau NumPy float32 de forme (224, 224).
    """
    if _is_dicom_bytes(image_bytes):
        gray = _decode_dicom_grayscale(image_bytes)
        img = Image.fromarray(gray, mode="L")
    else:
        try:
            img = Image.open(BytesIO(image_bytes)).convert("L")
        except Exception:
            gray = _decode_dicom_grayscale(image_bytes)
            img = Image.fromarray(gray, mode="L")

    img.thumbnail(IMG_SIZE, Image.BICUBIC)

    canvas = Image.new("L", IMG_SIZE, color=0)
    offset_x = (IMG_SIZE[0] - img.size[0]) // 2
    offset_y = (IMG_SIZE[1] - img.size[1]) // 2
    canvas.paste(img, (offset_x, offset_y))

    return np.array(canvas, dtype=np.float32)


def _extract_hog(img: np.ndarray) -> np.ndarray:
    """
    Étape HOG → vecteur de 576 features.
    Paramètres identiques au notebook d'entraînement du binôme.
    """
    return hog(img, **HOG_PARAMS, visualize=False)


def _extract_efficientnet_features(image_bytes: bytes) -> np.ndarray:
    """
    Extrait un embedding EfficientNet-B0 (taille 1280) depuis l'image source.

    Cette voie n'est utilisée que si l'artefact chargé indique feature_backend=efficientnet.
    """
    runtime = _load_efficientnet_extractor()
    if runtime is None:
        raise RuntimeError(
            "Le modèle EfficientNet est demandé mais PyTorch/torchvision ne sont pas disponibles."
        )

    torch = runtime["torch"]
    weights = runtime["weights"]
    extractor = runtime["extractor"]
    device = runtime["device"]

    preprocess = weights.transforms()
    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        features = extractor(tensor).flatten(1).cpu().numpy()

    return features[0]


# ── Interface publique ────────────────────────────────────────────────────────

def predict_image(image_bytes: bytes) -> Tuple[str, float, str]:
    """
    Pipeline complet : octets image → prédiction clinique.

    Raises:
        ValueError        : image rejetée par la garde OOD
        FileNotFoundError : modèle introuvable

    Returns:
        label       : 'PNEUMONIA' ou 'NORMAL'
        probability : score de confiance pour la classe PNEUMONIA [0.0 – 1.0]
        version     : identifiant de version du modèle
    """
    if not is_valid_xray(image_bytes):
        raise ValueError(
            "Format d'image invalide : soumettez une radiographie thoracique standard "
            "(JPEG ou PNG, incidence PA). Les images colorées ne sont pas acceptées."
        )

    pkg       = _load_model()
    model     = pkg['model']
    pca       = pkg.get('pca')
    label_map = pkg.get('label_map', {0: 'NORMAL', 1: 'PNEUMONIA'})
    model_version = pkg.get('model_version', MODEL_VERSION)
    feature_backend = pkg.get('feature_backend', 'hog')

    if feature_backend == 'efficientnet':
        features = _extract_efficientnet_features(image_bytes)
        if pca is not None:
            reduced = pca.transform(features.reshape(1, -1))[0]
        else:
            reduced = features
    else:
        img = _preprocess(image_bytes)
        features = _extract_hog(img)
        if pca is not None:
            reduced = pca.transform(features.reshape(1, -1))[0]
        else:
            reduced = features

    pred_class     = int(model.predict([reduced])[0])
    probas         = model.predict_proba([reduced])[0]
    pneumonia_idx  = list(model.classes_).index(1)
    probability    = float(probas[pneumonia_idx])

    return label_map.get(pred_class, 'NORMAL'), probability, model_version


def generate_heatmap(image_bytes: bytes, output_path: str) -> None:
    """
    Génère une visualisation côte-à-côte : image originale | gradients HOG colorisés.
    Permet de visualiser les zones de l'image ayant contribué à la prédiction.
    """
    img  = _preprocess(image_bytes)
    gray = img.astype(np.uint8)

    _, hog_image = hog(img, **HOG_PARAMS, visualize=True)

    hog_norm    = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hog_colored = cv2.applyColorMap(hog_norm, cv2.COLORMAP_JET)
    gray_bgr    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay     = cv2.addWeighted(gray_bgr, 0.6, hog_colored, 0.4, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gray_bgr, "Original",  (8, 22), font, 0.6, (200, 200, 200), 1)
    cv2.putText(overlay,  "Gradients", (8, 22), font, 0.6, (200, 200, 200), 1)

    separator = np.full((IMG_SIZE[1], 3, 3), 160, dtype=np.uint8)
    combined  = np.hstack([gray_bgr, separator, overlay])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, combined)


def anonymize_image(image_bytes: bytes) -> bytes:
    """
    Anonymise une radiographie par masquage noir irréversible des coins.

    Justification : les systèmes PACS inscrivent les données nominatives du patient
    (nom, date de naissance, numéro d'identification) dans les coins de l'image.
    Un masque noir opaque est plus robuste qu'un floutage, qui peut être inversé.

    Zones masquées : 6 % de la hauteur × 25 % de la largeur dans chaque coin.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            img = _decode_dicom_grayscale(image_bytes)
        except Exception:
            return image_bytes
    if img is None:
        return image_bytes

    h, w   = img.shape
    mask_h = int(h * 0.06)
    mask_w = int(w * 0.25)

    img[:mask_h,  :mask_w]  = 0
    img[:mask_h,  -mask_w:] = 0
    img[-mask_h:, :mask_w]  = 0
    img[-mask_h:, -mask_w:] = 0

    success, encoded = cv2.imencode(".png", img)
    return encoded.tobytes() if success else image_bytes
