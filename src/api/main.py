from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import trajets, gares, trains, operateurs, routes, stats
from src.api.routers import ml
from src.api.ml import models as ml_models
from src.api.auth import verify_token          # ← new


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models.load_models()
    yield

app = FastAPI(
    title="API Data Ferroviaires",
    description="""API REST pour consulter les dessertes ferroviaires du data warehouse.

La documentation interactive Swagger est disponible sur `/docs` et permet de tester les
endpoints protégés par authentification via le bouton Authorize.
""",
    version="1.0.0",
    root_path="/api",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ✅ All routes under these routers are now protected
app.include_router(trajets.router,    prefix="/trajets",
                   tags=["Trajets"],  dependencies=[Depends(verify_token)])
app.include_router(gares.router,      prefix="/gares",
                   tags=["Gares"],             dependencies=[Depends(verify_token)])
app.include_router(trains.router,     prefix="/trains",
                   tags=["Trains"],            dependencies=[Depends(verify_token)])
app.include_router(operateurs.router, prefix="/operateurs",
                   tags=["Opérateurs"],        dependencies=[Depends(verify_token)])
app.include_router(routes.router,     prefix="/routes",
                   tags=["Routes"],            dependencies=[Depends(verify_token)])
app.include_router(stats.router,      prefix="/stats",
                   tags=["Statistiques"],      dependencies=[Depends(verify_token)])
app.include_router(ml.router,         prefix="/ml",
                   tags=["Machine Learning"],  dependencies=[Depends(verify_token)])

# ✅ Public routes — no token required


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
