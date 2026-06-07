from pydantic import BaseModel, Field
from typing import Optional


class RelationInput(BaseModel):
    """Input schema for a single railway relation — mirrors the ETL CSV columns."""
    data_source:            str = Field(..., json_schema_extra={"example": "austria_etl"})
    route_id:               str = Field(..., json_schema_extra={"example": "AT-001"})
    id_origin_city:         str = Field(..., json_schema_extra={"example": "Vienna"})
    id_destination_city:    str = Field(..., json_schema_extra={"example": "Salzburg"})
    weekly_train:           int = Field(..., ge=0, json_schema_extra={"example": 3})

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data_source":         "austria_etl",
                    "route_id":            "AT-001",
                    "id_origin_city":      "Vienna",
                    "id_destination_city": "Salzburg",
                    "weekly_train":        3,
                }
            ]
        }
    }


class ClassificationResult(BaseModel):
    """Result of the Random Forest classifier."""
    desserte_type: str              = Field(...)
    probabilities: dict[str, float] = Field(...)
    model_used:    str              = Field(...)

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "desserte_type": "Sous-desservi",
                "probabilities": {
                    "Bien desservi":    0.0,
                    "Desserte Normale": 0.12,
                    "Sous-desservi":    0.88,
                },
                "model_used": "RandomForest",
            }
        },
    }


class RetrainResponse(BaseModel):
    """Returned immediately when a background retrain is triggered."""
    status:  str = Field(...)
    message: str = Field(...)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status":  "accepted",
                "message": "Retraining started in background.",
            }
        }
    }
