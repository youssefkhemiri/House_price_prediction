#!/usr/bin/env python3
"""
Pytest tests for the Real Estate Scraper FastAPI API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app, SUPPORTED_PROVIDERS

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)

@pytest.fixture
def mock_property_data():
    """Sample property data for testing"""
    return {
        "url": "https://www.menzili.tn/test-property",
        "id": "test-123",
        "source": "menzili",
        "property_type": "Maison",
        "price": "250000",
        "currency": "DT",
        "description": ["Beautiful house with garden"],
        "address": "Tunis, Tunisia",
        "governorate": "Tunis",
        "delegation": "Carthage",
        "locality": "Sidi Bou Said",
        "living_area": "150",
        "land_area": 300,
        "room_count": 4,
        "bathroom_count": 2,
        "has_garage": True,
        "has_garden": True,
        "has_pool": False,
        "photos": ["photo1.jpg", "photo2.jpg"],
        "features": ["garage", "garden"],
        "transaction_type": "Vente"
    }

@pytest.fixture
def mock_prediction_data():
    """Sample prediction data for testing"""
    return {
        "predicted_price": 275000,
        "confidence": 0.85,
        "price_range": {"min": 250000, "max": 300000},
        "factors": ["location", "size", "features"]
    }

class TestAPIEndpoints:
    """Test class for API endpoints"""

    def test_api_scrape_valid_menzili(self, client, mock_property_data, mock_prediction_data):
        """Test API scraping with valid Menzili URL"""
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = mock_property_data
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"] == mock_property_data
            assert data["prediction"] == mock_prediction_data
            assert data["error"] is None

    def test_api_scrape_valid_mubawab(self, client, mock_property_data, mock_prediction_data):
        """Test API scraping with valid Mubawab URL"""
        mock_property_data["source"] = "mubawab"
        
        with patch('app.scrape_mubawab') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = mock_property_data
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/api/scrape", json={
                "provider": "mubawab",
                "url": "https://www.mubawab.tn/test-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["source"] == "mubawab"
            assert data["prediction"] == mock_prediction_data

    def test_api_scrape_any_website(self, client, mock_property_data, mock_prediction_data):
        """Test API scraping with any website"""
        mock_property_data["source"] = "any_website"
        
        with patch('app.scrape_any_website') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = mock_property_data
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/api/scrape", json={
                "provider": "any_website",
                "url": "https://example.com/property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["source"] == "any_website"

    def test_api_scrape_invalid_provider(self, client):
        """Test API scraping with invalid provider"""
        response = client.post("/api/scrape", json={
            "provider": "invalid_provider",
            "url": "https://example.com/property"
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid provider" in data["detail"]

    def test_api_scrape_empty_url(self, client):
        """Test API scraping with empty URL"""
        response = client.post("/api/scrape", json={
            "provider": "menzili",
            "url": ""
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "URL is required" in data["detail"]

    def test_api_scrape_scraper_exception(self, client):
        """Test API scraping when scraper raises exception"""
        with patch('app.scrape_menzili') as mock_scrape:
            mock_scrape.side_effect = Exception("Scraping failed")
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Scraping failed" in data["error"]

    def test_api_scrape_prediction_failure(self, client, mock_property_data):
        """Test API scraping when prediction fails"""
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = mock_property_data
            mock_predict.side_effect = Exception("Prediction failed")
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"] == mock_property_data
            assert data["prediction"] is None  # Prediction should be None when it fails

    def test_api_predict_valid_data(self, client, mock_property_data, mock_prediction_data):
        """Test API prediction with valid property data"""
        with patch('app.predict_price') as mock_predict:
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/api/predict", json={
                "property_data": mock_property_data
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["prediction"] == mock_prediction_data
            assert data["error"] is None

    def test_api_predict_invalid_data(self, client):
        """Test API prediction with invalid property data"""
        with patch('app.predict_price') as mock_predict:
            mock_predict.side_effect = Exception("Invalid data")
            
            response = client.post("/api/predict", json={
                "property_data": {}
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "Invalid data" in data["error"]

    def test_api_predict_missing_data(self, client):
        """Test API prediction with missing property_data field"""
        response = client.post("/api/predict", json={})
        
        assert response.status_code == 422  # Validation error

class TestWebEndpoints:
    """Test class for web interface endpoints"""

    def test_index_page(self, client):
        """Test main index page loads"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Real Estate Scraper" in response.text

    def test_manual_entry_page(self, client):
        """Test manual entry page loads"""
        response = client.get("/manual")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Manual Property Entry" in response.text

    def test_docs_page(self, client):
        """Test API docs page loads"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_scrape_form_valid_submission(self, client, mock_property_data, mock_prediction_data):
        """Test web form scrape submission with valid data"""
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = mock_property_data
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/scrape", data={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test-property"
            })
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    # def test_scrape_form_invalid_provider(self, client):
    #     """Test web form scrape with invalid provider"""
    #     response = client.post("/scrape", data={
    #         "provider": "invalid_provider",
    #         "url": "https://example.com/property"
    #     })
        
    #     assert response.status_code == 200
    #     assert "Invalid provider" in response.text

    def test_manual_entry_form_submission(self, client, mock_prediction_data):
        """Test manual entry form submission"""
        with patch('app.predict_price') as mock_predict:
            mock_predict.return_value = mock_prediction_data
            
            response = client.post("/manual-entry", data={
                "property_type": "Maison",
                "transaction_type": "Vente",
                "price": "250000",
                "currency": "DT",
                "address": "Test Address",
                "living_area": "150",
                "room_count": "4"
            })
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

class TestDataValidation:
    """Test class for data validation"""

    def test_scrape_request_validation(self, client):
        """Test ScrapeRequest model validation"""
        # Missing provider
        response = client.post("/api/scrape", json={
            "url": "https://example.com"
        })
        assert response.status_code == 422

        # Missing URL
        response = client.post("/api/scrape", json={
            "provider": "menzili"
        })
        assert response.status_code == 422

        # Invalid JSON
        response = client.post("/api/scrape", data="invalid json")
        assert response.status_code == 422

    def test_prediction_request_validation(self, client):
        """Test PredictionRequest model validation"""
        # Missing property_data
        response = client.post("/api/predict", json={})
        assert response.status_code == 422

        # Invalid JSON
        response = client.post("/api/predict", data="invalid json")
        assert response.status_code == 422

class TestConfiguration:
    """Test application configuration"""

    def test_supported_providers(self):
        """Test that all supported providers are configured correctly"""
        assert isinstance(SUPPORTED_PROVIDERS, dict)
        assert len(SUPPORTED_PROVIDERS) > 0
        
        for provider_id, provider_info in SUPPORTED_PROVIDERS.items():
            assert isinstance(provider_id, str)
            assert isinstance(provider_info, dict)
            assert "name" in provider_info
            assert "domain" in provider_info
            assert "description" in provider_info
            assert "icon" in provider_info

    def test_app_metadata(self):
        """Test FastAPI app metadata"""
        assert app.title == "Real Estate Scraper"
        assert app.description == "Extract property data from real estate websites"
        assert app.version == "1.0.0"

class TestErrorHandling:
    """Test error handling scenarios"""

    def test_network_timeout_simulation(self, client):
        """Test handling of network timeouts"""
        with patch('app.scrape_menzili') as mock_scrape:
            mock_scrape.side_effect = TimeoutError("Request timed out")
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "timed out" in data["error"].lower()

    def test_invalid_url_format(self, client):
        """Test handling of invalid URL formats"""
        response = client.post("/api/scrape", json={
            "provider": "menzili",
            "url": "not-a-valid-url"
        })
        
        # Should still attempt to scrape but likely fail
        assert response.status_code in [200, 400]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
