from fastapi import APIRouter, HTTPException, BackgroundTasks

from src.api.ml.schemas import (
    RelationInput,
    ClassificationResult,
    RetrainResponse,
)
from src.api.ml import models as ml_models

router = APIRouter()


@router.post(
    "/classify",
    response_model=ClassificationResult,
    summary="Classifier une relation ferroviaire",
    description=(
        "Prédit le **desserte_type** d'une relation ferroviaire "
        "(`Bien desservi`, `Desserte Normale`, `Sous-desservi`) "
        "à partir de ses caractéristiques.\n\n"
        "Modèle : **Random Forest** (200 arbres, class_weight=balanced) "
        "entraîné sur les données ETL des 3 pays (Autriche, Allemagne, Espagne)."
    ),
)
def classify_relation(payload: RelationInput) -> ClassificationResult:
    try:
        result = ml_models.predict_desserte(payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
    return ClassificationResult.model_validate(result)


@router.post(
    "/retrain",
    response_model=RetrainResponse,
    status_code=202,
    summary="Réentraîner les modèles",
    description=(
        "Relance l'entraînement du modèle de **classification** "
        "en tâche de fond à partir des fichiers CSV dans `/data/output`.\n\n"
        "Les nouveaux modèles sont sauvegardés dans `src/training/model/`. "
        "Retourne immédiatement **202 Accepted**."
    ),
)
def retrain(background_tasks: BackgroundTasks) -> RetrainResponse:
    background_tasks.add_task(ml_models.train_and_save)
    return RetrainResponse(
        status="accepted",
        message=(
            f"Retraining started in background. "
            f"Models will be saved to: {ml_models.MODEL_FOLDER}"
        ),
    )
