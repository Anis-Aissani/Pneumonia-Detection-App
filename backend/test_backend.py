"""
test_backend.py — Suite de tests unitaires pour PneumoScan
Exécution : pytest test_backend.py -v

Organisation :
  TestValidation    — règles d'acceptation/rejet des fichiers entrants
  TestPreprocessing — invariants du pipeline de prétraitement d'images
  TestHOGFeatures   — propriétés du vecteur de features HOG
  TestTriageLogic   — règles métier de classification des priorités
  TestAnonymization — comportement de l'anonymisation des marges
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock

from database import compute_triage
from predictor import _extract_hog, _preprocess, anonymize_image
from utils import validate_image


# ── Helpers de test ───────────────────────────────────────────────────────────

def make_image(mode="RGB", size=(300, 300), color=128, fmt="JPEG") -> bytes:
    """Génère une image synthétique en mémoire pour les tests."""
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def mock_file(content_type: str, filename: str) -> MagicMock:
    """Simule un objet UploadFile FastAPI pour les tests de validation."""
    f = MagicMock()
    f.content_type = content_type
    f.filename     = filename
    return f


# ── 1. Validation des fichiers entrants ──────────────────────────────────────

class TestValidation:
    """
    Vérifie que validate_image() accepte les formats JPEG, PNG et DICOM,
    en contrôlant à la fois le type MIME et l'extension du fichier.
    """

    def test_accepte_jpeg(self):
        assert validate_image(mock_file("image/jpeg", "scan.jpg")) is True

    def test_accepte_png(self):
        assert validate_image(mock_file("image/png", "scan.png")) is True

    def test_accepte_dicom(self):
        assert validate_image(mock_file("application/dicom", "scan.dcm")) is True

    def test_rejette_pdf(self):
        assert validate_image(mock_file("application/pdf", "doc.pdf")) is False

    def test_rejette_gif(self):
        assert validate_image(mock_file("image/gif", "anim.gif")) is False

    def test_rejette_extension_incorrecte(self):
        """MIME valide mais extension incorrecte : les deux vérifications doivent échouer."""
        assert validate_image(mock_file("image/jpeg", "scan.bmp")) is False

    def test_rejette_texte_deguise(self):
        """Un fichier texte avec extension .jpg ne doit pas être accepté."""
        assert validate_image(mock_file("text/plain", "fake.jpg")) is False


# ── 2. Prétraitement des images ───────────────────────────────────────────────

class TestPreprocessing:
    """
    Vérifie les invariants de _preprocess() : quelle que soit l'image en entrée,
    la sortie doit toujours être un tableau NumPy float32 de forme (224, 224).
    """

    @pytest.mark.parametrize("size", [(300, 300), (800, 200), (10, 10), (2000, 2000)])
    def test_sortie_toujours_224x224(self, size):
        """Le canvas de sortie est 224×224 quelle que soit la taille ou le ratio d'entrée."""
        assert _preprocess(make_image(size=size)).shape == (224, 224)

    def test_type_float32(self):
        assert _preprocess(make_image()).dtype == np.float32

    def test_image_noire_reste_noire(self):
        """Une image entièrement noire ne doit pas introduire de signal artificiellement."""
        assert _preprocess(make_image(color=0)).max() == 0.0

    def test_accepte_format_png(self):
        assert _preprocess(make_image(fmt="PNG")).shape == (224, 224)


# ── 3. Extraction des features HOG ───────────────────────────────────────────

class TestHOGFeatures:
    """
    Vérifie les propriétés mathématiques du vecteur HOG.
    Dimension attendue : 7 × 7 × (2×2) × 4 = 576 features.
    Le vecteur ne doit contenir aucune valeur NaN ou infinie.
    """

    def test_dimension_576_features(self):
        assert _extract_hog(_preprocess(make_image())).shape == (576,)

    def test_image_noire_donne_gradient_nul(self):
        """Une image uniforme produit un gradient nul : le vecteur HOG est entièrement zéro."""
        assert np.all(_extract_hog(_preprocess(make_image(color=0))) == 0)

    def test_stabilite_numerique(self):
        """Le vecteur doit être numériquement stable (pas de NaN ni d'infini) pour le SVM."""
        assert np.all(np.isfinite(_extract_hog(_preprocess(make_image()))))


# ── 4. Logique de triage ──────────────────────────────────────────────────────

class TestTriageLogic:
    """
    Vérifie les règles de triage clinique définies dans database.py :
      CRITICAL : pneumonie avec probabilité > 85%
            MODERATE : pneumonie avec probabilité entre 15% et 85%
      ROUTINE  : résultat normal (quelle que soit la probabilité)
    """

    @pytest.mark.parametrize("prob", [0.86, 0.90, 0.99])
    def test_pneumonie_haute_confiance_est_critique(self, prob):
        assert compute_triage("PNEUMONIA", prob) == "CRITICAL"

    @pytest.mark.parametrize("prob", [0.16, 0.50, 0.84, 0.85])
    def test_pneumonie_confiance_moyenne_est_moderee(self, prob):
        """Le seuil de 0.85 est inclus dans MODERATE (opérateur strict >)."""
        assert compute_triage("PNEUMONIA", prob) == "MODERATE"

    @pytest.mark.parametrize("prob", [0.01, 0.14])
    def test_pneumonie_faible_confiance_est_routine(self, prob):
        assert compute_triage("PNEUMONIA", prob) == "ROUTINE"

    @pytest.mark.parametrize("prob", [0.01, 0.50, 0.99])
    def test_normal_toujours_routine(self, prob):
        """Un résultat NORMAL est toujours ROUTINE, quel que soit le score."""
        assert compute_triage("NORMAL", prob) == "ROUTINE"


# ── 5. Anonymisation ──────────────────────────────────────────────────────────

class TestAnonymization:
    """
    Vérifie que anonymize_image() floute bien les marges de l'image
    et retourne des octets valides dans tous les cas, y compris les cas limites.
    """

    def test_retourne_des_bytes_valides(self):
        result = anonymize_image(make_image())
        assert isinstance(result, bytes) and len(result) > 0

    def test_les_bords_superieurs_sont_modifies(self):
        """Après anonymisation, les premières lignes de pixels doivent différer de l'original."""
        original   = make_image(color=200)
        anonymized = anonymize_image(original)

        orig_arr = cv2.imdecode(np.frombuffer(original,   dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        anon_arr = cv2.imdecode(np.frombuffer(anonymized, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        assert not np.array_equal(orig_arr[:5, :], anon_arr[:5, :])

    def test_image_invalide_retournee_sans_exception(self):
        """Des octets corrompus doivent être retournés tels quels sans lever d'exception."""
        bad_bytes = b"not_an_image"
        assert anonymize_image(bad_bytes) == bad_bytes
