# 🫁 PneumoScan — Plateforme de Diagnostic de Pneumonie

Système d'aide au diagnostic de pneumonie intégrant un modèle HOG dans une architecture logicielle complète.

---

## 🗂️ Structure du Projet

```
pneumonia-platform/
├── backend/
│   ├── main.py          ← FastAPI app (endpoints /predict, /history, /heatmap)
│   ├── predictor.py     ← Intégration modèle HOG + génération heatmap
│   ├── database.py      ← SQLite logging (traçabilité)
│   ├── scanner.py       ← Module scan automatique /incoming_scans
│   ├── utils.py         ← Validation fichiers
│   ├── model/           ← ⚠️ Mettre votre .pkl ici
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py           ← Interface Streamlit
│   ├── requirements.txt
│   └── Dockerfile
├── incoming_scans/      ← Dépôt automatique de radiographies
├── heatmaps/            ← Heatmaps générées
├── data/                ← Base SQLite
└── docker-compose.yml
```

---

## 🚀 Démarrage Rapide

### 1. Ajouter votre modèle

Placez le fichier `.pkl` de votre modèle HOG ici :

```
backend/model/pneumonia_hog_model.pkl
```

Le modèle doit être un classificateur sklearn (`predict()` + `predict_proba()` ou `decision_function()`).

### 2. Ajuster les paramètres HOG (si nécessaire)

Dans `backend/predictor.py`, vérifiez que ces paramètres correspondent à l'entraînement de votre ami :

```python
HOG_WIN_SIZE  = (128, 128)   # Taille d'entrée
HOG_CELL_SIZE = (8, 8)
HOG_BLOCK_SIZE = (16, 16)
HOG_NBINS     = 9
```

### 3. Lancer avec Docker

```bash
docker-compose up --build
```

| Service  | URL                        |
|----------|----------------------------|
| Frontend | http://localhost:8501      |
| API      | http://localhost:8000      |
| API Docs | http://localhost:8000/docs |

### 4. Développement local (sans Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (autre terminal)
cd frontend
pip install -r requirements.txt
# Changer API_URL dans app.py: "http://localhost:8000"
streamlit run app.py
```

---

## 📡 API Endpoints

| Méthode | Endpoint         | Description                     |
|---------|------------------|---------------------------------|
| POST    | `/predict`       | Analyse une radiographie        |
| GET     | `/history`       | Historique des prédictions      |
| GET     | `/heatmap/{id}`  | Récupère une heatmap            |
| GET     | `/health`        | Statut de l'API                 |
| GET     | `/docs`          | Documentation Swagger auto      |

### Exemple `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg"
```

Réponse :
```json
{
  "id": "uuid",
  "prediction": "PNEUMONIA",
  "probability": 0.87,
  "model_version": "1.0.0-HOG",
  "heatmap_url": "/heatmap/heatmap_xxx.png",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🏥 Module Scan Automatique

Déposez des radiographies dans `incoming_scans/` et le système les traitera automatiquement toutes les 10 secondes. Les fichiers traités sont déplacés dans `incoming_scans/processed/`.

---

## 🗃️ Base de Données

Table `predictions` (SQLite par défaut, PostgreSQL possible) :

| Champ          | Type    | Description               |
|----------------|---------|---------------------------|
| id             | TEXT PK | UUID unique               |
| timestamp      | TEXT    | ISO 8601                  |
| image_name     | TEXT    | Nom du fichier            |
| prediction     | TEXT    | PNEUMONIA / NORMAL        |
| probability    | REAL    | Score entre 0 et 1        |
| model_version  | TEXT    | Version du modèle         |
| heatmap_path   | TEXT    | Chemin de la heatmap      |

---

## ⚠️ Avertissement Médical

Ce système est un **prototype académique** et ne remplace pas un diagnostic médical professionnel.
