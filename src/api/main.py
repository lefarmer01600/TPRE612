from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import trajets, gares, trains, operateurs, routes, stats
from src.api.routers import ml                  # ← new
from src.api.ml import models as ml_models      # ← new


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load (or train from scratch) ML models once at startup."""
    ml_models.load_models()
    yield
    # nothing to clean up — joblib models are stateless


app = FastAPI(
    title="API Data Ferroviaires",
    description="""
API REST – Data Ferroviaires

Fonctionnalités
- Consultation des trajets avec filtres multicritères
- Recherche de gares par ville, pays
- Statistiques : émissions CO2, fréquentation, performance
- Compatible Grafana (JSON datasource)
- **ML** : classification et clustering des dessertes ferroviaires
    """,
    version="1.0.0",
    root_path="/api",
    lifespan=lifespan,          # replaces deprecated @app.on_event
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # POST required for /ml/* routes
    allow_headers=["*"],
)

app.include_router(trajets.router,    prefix="/trajets",    tags=["Trajets"])
app.include_router(gares.router,      prefix="/gares",      tags=["Gares"])
app.include_router(trains.router,     prefix="/trains",     tags=["Trains"])
app.include_router(operateurs.router, prefix="/operateurs", tags=["Opérateurs"])
app.include_router(routes.router,     prefix="/routes",     tags=["Routes"])
app.include_router(stats.router,      prefix="/stats",      tags=["Statistiques"])
app.include_router(ml.router,         prefix="/ml",         tags=["Machine Learning"])


@app.get("/", tags=["Santé"])
def root():
    return {
        "status": "ok",
        "message": "API Dessertes Ferroviaires v1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Santé"])
def health():
    return {"status": "healthy"}
