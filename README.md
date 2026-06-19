# TPRE612 - Data Warehouse & IA Pipeline

## Description

**TPRE612** est un projet de **Data Warehouse** combinant l'ETL (Extract, Transform, Load) de données de transports ferroviaires, une **API REST** modernes avec FastAPI et des modèles de **Machine Learning** pour l'analyse et prédiction.

Le projet agrège des données multi-sources (SNCF, Eurostat, OpenData) et expose une API pour l'accès aux données traitées et les prédictions du modèle d'IA.

---

## Architecture du Projet

```
TPRE612/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── schemas/
│   │   ├── db/
│   │   ├── models/
│   │   └── README.md
│   │
│   ├── etl/
│   │   ├── gestion_etl.py
│   │   ├── eurostat.py
│   │   ├── sncf.py
│   │   ├── CO2.py
│   │   ├── data_gouv.py
│   │   ├── dataeuropa.py
│   │   ├── night_train_data.py
│   │   ├── open-data.ipynb
│   │   └── populate_data_warehouse/
│   │
│   └── training/
│       ├── etl/
│       └── model/
│
├── data/
│   ├── eurostat_rail_data.csv
│   ├── austria/
│   ├── germany/
│   ├── spain/
│   ├── husahuc_data_france/
│   ├── night-train-data/
│   ├── europa.eu/
│   ├── opendata/
│   └── output/
│
├── CREATE_TPRE612_DATA_WAREHOUSE.sql
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .gitlab-ci.yml
├── MCD.loo
├── export_first_rows_from_tables.py
└── README.md
```

---

## Stack Technique

### Backend & API
- **FastAPI** 0.115.0 - Framework API moderne et performant
- **Uvicorn** 0.30.6 - Serveur ASGI
- **Pydantic** 2.9.2 - Validation des données
- **SQLAlchemy** 2.0.48 - ORM pour la base de données
- **psycopg2** 2.9.9 - Driver PostgreSQL

### Data Science & IA
- **scikit-learn** 1.4.0 - Modèles ML (regression, classification, clustering)
- **pandas** 3.0.1 - Manipulation des données
- **numpy** 1.26.0 - Opérations matricielles
- **joblib** 1.3.0 - Sérialisation des modèles
- **missingno** 0.5.2 - Analyse des données manquantes

### Infrastructure
- **Docker** - Containerisation
- **PostgreSQL** - Base de données
- **Python** 3.11.14+ - Runtime

### Développement
- **pytest** 8.0 - Tests unitaires
- **pytest-asyncio** 0.23 - Tests asynchrones
- **ruff** 0.4 - Linting et formatage
- **httpx** 0.27 - Client HTTP pour tests

---

## Installation

### Prérequis
- Python 3.11.14+
- PostgreSQL 12+
- Docker (optionnel)

### Cloner le projet
```bash
git clone <repository-url>
cd TPRE612
```

### Installer les dépendances
```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip classique
pip install -e ".[dev]"
```

### Configurer la base de données
```bash
# Créer la base de données
psql -U postgres -f CREATE_TPRE612_DATA_WAREHOUSE.sql

# Ou via SQLAlchemy
python -m src.api.db.init
```

### Configurer l'environnement
```bash
# Créer un fichier .env
cat > .env << EOF
DATABASE_URL=postgresql://user:password@localhost:5432/tpre612
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000
EOF
```

---

## Utilisation

### Démarrer l'API
```bash
# Développement avec rechargement automatique
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

L'API sera disponible sur `http://localhost:8000`
- Documentation : `http://localhost:8000/docs`

---

## Modèle d'IA

### Type de modèle
- **Algorithme** : scikit-learn (Classification)
- **Framework** : scikit-learn + numpy + pandas
- **Sauvegarde** : joblib (fichier `.pkl`)

### Pipeline d'entraînement
```
Données brutes
    ↓
Nettoyage & Préparation
    ↓
Feature Engineering
    ↓
Entraînement du modèle
    ↓
Validation & Métriques
    ↓
Sérialisation (joblib)
    ↓
Intégration dans l'API
```