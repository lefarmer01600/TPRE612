from contextlib import asynccontextmanager
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import trajets, gares, trains, operateurs, routes, stats
from src.api.routers import ml
from src.api.ml import models as ml_models
from src.api.auth import verify_token          # ← new
from src.api.error_handlers import register_error_handlers
from src.api.logging_config import setup_logging

from fastapi.responses import JSONResponse
from sqlalchemy import text
from src.api.db.database import engine



setup_logging()
request_logger = logging.getLogger("api.request")


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

register_error_handlers(app)


@app.middleware("http")
async def structured_request_logging(request: Request, call_next):
    start_time = time.perf_counter()
    request_start = datetime.now(timezone.utc).isoformat()
    endpoint = request.url.path
    client_ip = request.client.host if request.client else None

    request_logger.debug(
        "request_started",
        extra={
            "timestamp": request_start,
            "endpoint": endpoint,
            "http_method": request.method,
            "client_ip": client_ip,
        },
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        request_logger.error(
            "request_failed",
            exc_info=True,
            extra={
                "timestamp": request_start,
                "endpoint": endpoint,
                "http_method": request.method,
                "duration_ms": duration_ms,
                "status_code": 500,
                "client_ip": client_ip,
            },
        )
        raise

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    status_code = response.status_code
    level = logging.INFO
    if status_code >= 500:
        level = logging.ERROR
    elif status_code >= 400:
        level = logging.WARNING

    request_logger.log(
        level,
        "request_completed",
        extra={
            "timestamp": request_start,
            "endpoint": endpoint,
            "http_method": request.method,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "client_ip": client_ip,
        },
    )

    return response

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


@app.get("/health/ready", tags=["Santé"])
def readiness():
    """Readiness probe : vérifie la connexion à PostgreSQL (standalone)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "database": "up"}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "database": "down", "detail": str(exc)},
        )
