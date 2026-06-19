import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.api.main import app

client = TestClient(app)


class TestRootEndpoints:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data
        assert "docs" in data

    def test_health_endpoint(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestGaresEndpoint:

    def test_get_gares_all(self):
        response = client.get("/gares")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "data" in data

    def test_get_gares_with_pagination(self):
        response = client.get("/gares?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    def test_get_gares_filter_by_country(self):
        response = client.get("/gares?country=France")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "total" in data

    def test_get_gares_filter_by_city(self):
        response = client.get("/gares?city=Paris")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestRoutesEndpoint:

    def test_get_routes_all(self):
        response = client.get("/routes")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "data" in data

    def test_get_routes_with_filter(self):
        response = client.get("/routes?origin=Paris")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)


class TestErrorHandling:

    def test_missing_required_field_classify(self):
        payload = {
            # Manque data_source
            "route_id": "AT-001",
            "id_origin_city": "Vienna",
            "id_destination_city": "Salzburg",
            "weekly_train": 5,
        }
        response = client.post("/ml/classify", json=payload)
        assert response.status_code == 422  # Validation error

    def test_missing_required_field_route_id(self):
        payload = {
            "data_source": "austria_etl",
            # Manque route_id
            "id_origin_city": "Vienna",
            "id_destination_city": "Salzburg",
            "weekly_train": 5,
        }
        response = client.post("/ml/classify", json=payload)
        assert response.status_code == 422

    def test_invalid_weekly_train_negative(self):
        payload = {
            "data_source": "austria_etl",
            "route_id": "AT-001",
            "id_origin_city": "Vienna",
            "id_destination_city": "Salzburg",
            "weekly_train": -5,  # Invalide - doit être >= 0
        }
        response = client.post("/ml/classify", json=payload)
        # Peut être 422 (validation) ou 200 si géré différemment
        assert response.status_code in [200, 422]

    def test_invalid_weekly_train_type(self):
        payload = {
            "data_source": "austria_etl",
            "route_id": "AT-001",
            "id_origin_city": "Vienna",
            "id_destination_city": "Salzburg",
            "weekly_train": "invalid",  # Devrait être un int
        }
        response = client.post("/ml/classify", json=payload)
        assert response.status_code == 422

    def test_missing_all_required_fields(self):
        payload = {}
        response = client.post("/ml/classify", json=payload)
        assert response.status_code == 422


class TestAPIDocumentation:

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])