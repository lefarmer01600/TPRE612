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


class ClusterProfile(BaseModel):
    """Descriptive profile attached to a KMeans cluster."""
    size:                int
    weekly_train_mean:   float
    weekly_train_median: float
    weekly_train_std:    float
    desserte_type_dist:  dict[str, float]

    model_config = {"from_attributes": True}


class ClusteringResult(BaseModel):
    """Result of the KMeans clustering."""
    cluster_id:      int
    cluster_profile: Optional[ClusterProfile]
    model_used:      str = Field(...)

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "cluster_id":      1,
                "cluster_profile": None,
                "model_used":      "KMeans",
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
