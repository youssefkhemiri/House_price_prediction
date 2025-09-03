#!/usr/bin/env python3
"""
Integration tests for the Real Estate Scraper API
These tests check the complete workflow from request to response
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import time
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import app

@pytest.fixture
def client():
    """Create a test client for integration tests"""
    return TestClient(app)

@pytest.mark.integration
class TestFullAPIWorkflow:
    """Integration tests for complete API workflows"""

    @patch('app.scrape_menzili')
    @patch('app.predict_price')
    def test_complete_scraping_workflow(self, mock_predict, mock_scrape, client, sample_property_data, sample_prediction_data):
        """Test complete workflow: scrape -> predict -> return results"""
        
        # Setup mocks
        mock_scrape.return_value = sample_property_data
        mock_predict.return_value = sample_prediction_data
        
        # Make API request
        response = client.post("/api/scrape", json={
            "provider": "menzili",
            "url": "https://www.menzili.tn/test-property"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"] == sample_property_data
        assert data["prediction"] == sample_prediction_data
        assert data["error"] is None
        
        # Verify mocks were called correctly
        mock_scrape.assert_called_once_with("https://www.menzili.tn/test-property")
        mock_predict.assert_called_once_with(sample_property_data)

    @patch('app.predict_price')
    def test_manual_entry_to_prediction_workflow(self, mock_predict, client, sample_prediction_data):
        """Test manual entry workflow: form submit -> predict -> display results"""
        
        mock_predict.return_value = sample_prediction_data
        
        # Submit manual entry form
        response = client.post("/manual-entry", data={
            "property_type": "Villa",
            "transaction_type": "Vente",
            "price": "300000",
            "currency": "DT",
            "address": "Sidi Bou Said, Tunis",
            "governorate": "Tunis",
            "living_area": "200",
            "room_count": "4",
            "bathroom_count": "3",
            "has_garage": "on",  # Form checkboxes send "on" when checked
            "has_pool": "on"
        })
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Verify prediction was called
        mock_predict.assert_called_once()
        
        # Check that the call was made with properly formatted data
        call_args = mock_predict.call_args[0][0]  # First argument of the call
        assert call_args["property_type"] == "Villa"
        assert call_args["transaction_type"] == "Vente"
        assert call_args["has_garage"] is True
        assert call_args["has_pool"] is True

    def test_web_interface_to_api_consistency(self, client):
        """Test that web interface and API return consistent data structures"""
        
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            property_data = {
                "property_type": "Maison",
                "price": "250000",
                "source": "menzili"
            }
            prediction_data = {
                "predicted_price": 275000,
                "confidence": 0.85
            }
            
            mock_scrape.return_value = property_data
            mock_predict.return_value = prediction_data
            
            # Test API endpoint
            api_response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test"
            })
            
            # Test web form endpoint
            web_response = client.post("/scrape", data={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test"
            })
            
            # Both should succeed
            assert api_response.status_code == 200
            assert web_response.status_code == 200
            
            # API should return JSON
            api_data = api_response.json()
            assert api_data["data"] == property_data
            
            # Web should return HTML
            assert "text/html" in web_response.headers["content-type"]

@pytest.mark.integration
class TestErrorHandlingWorkflows:
    """Integration tests for error handling in complete workflows"""

    def test_scraping_error_propagation(self, client):
        """Test how scraping errors are handled through the API"""
        
        with patch('app.scrape_menzili') as mock_scrape:
            mock_scrape.side_effect = Exception("Network timeout")
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test"
            })
            
            assert response.status_code == 200  # API doesn't raise HTTP errors
            data = response.json()
            assert data["success"] is False
            assert "Network timeout" in data["error"]

    def test_prediction_error_handling(self, client, sample_property_data):
        """Test prediction errors don't break the scraping workflow"""
        
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = sample_property_data
            mock_predict.side_effect = Exception("Prediction model error")
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test"
            })
            
            # Scraping should still succeed even if prediction fails
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"] == sample_property_data
            assert data["prediction"] is None  # Prediction failed gracefully

    def test_validation_error_handling(self, client):
        """Test validation errors in API requests"""
        
        # Missing required fields
        response = client.post("/api/scrape", json={
            "provider": "menzili"
            # Missing URL
        })
        
        assert response.status_code == 422  # Validation error
        
        # Invalid provider
        response = client.post("/api/scrape", json={
            "provider": "nonexistent_provider",
            "url": "https://example.com"
        })
        
        assert response.status_code == 400  # Business logic error

@pytest.mark.integration
class TestCrossProviderConsistency:
    """Test that different providers return consistent data structures"""

    @patch('app.predict_price')
    def test_all_providers_return_consistent_structure(self, mock_predict, client):
        """Test that all providers return data in the same format"""
        
        mock_predict.return_value = {"predicted_price": 250000, "confidence": 0.8}
        
        providers_and_mocks = [
            ("menzili", "scrape_menzili"),
            ("mubawab", "scrape_mubawab"),
            ("any_website", "scrape_any_website")
        ]
        
        required_fields = ["source", "property_type", "url"]
        
        for provider, mock_func_name in providers_and_mocks:
            with patch(f'app.{mock_func_name}') as mock_scrape:
                
                # Create mock data with required fields
                mock_data = {
                    "url": f"https://example.com/{provider}",
                    "source": provider,
                    "property_type": "Maison",
                    "price": "250000"
                }
                mock_scrape.return_value = mock_data
                
                response = client.post("/api/scrape", json={
                    "provider": provider,
                    "url": f"https://example.com/{provider}"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                
                # Check that all required fields are present
                for field in required_fields:
                    assert field in data["data"], f"Missing {field} in {provider} response"

@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceAndReliability:
    """Integration tests for performance and reliability"""

    def test_concurrent_requests_handling(self, client):
        """Test that the API can handle concurrent requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            with patch('app.scrape_any_website') as mock_scrape:
                mock_scrape.return_value = {"source": "any_website", "property_type": "Test"}
                
                response = client.post("/api/scrape", json={
                    "provider": "any_website",
                    "url": "https://example.com/test"
                })
                results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        response_codes = []
        while not results.empty():
            response_codes.append(results.get())
        
        # All requests should succeed
        assert len(response_codes) == 5
        assert all(code == 200 for code in response_codes)

    def test_large_data_handling(self, client):
        """Test API handling of large property data"""
        
        # Create large mock data
        large_description = ["Very long description"] * 100
        large_features = [f"feature_{i}" for i in range(50)]
        large_photos = [f"https://example.com/photo_{i}.jpg" for i in range(20)]
        
        large_property_data = {
            "source": "menzili",
            "property_type": "Villa",
            "description": large_description,
            "features": large_features,
            "photos": large_photos,
            "url": "https://example.com/large-property"
        }
        
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = large_property_data
            mock_predict.return_value = {"predicted_price": 500000, "confidence": 0.9}
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://example.com/large-property"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["data"]["description"]) == 100
            assert len(data["data"]["features"]) == 50
            assert len(data["data"]["photos"]) == 20

@pytest.mark.integration
class TestAPIDocumentationConsistency:
    """Test that API behavior matches its documentation"""

    def test_openapi_schema_accessibility(self, client):
        """Test that OpenAPI schema is accessible and valid"""
        
        # Test docs endpoints
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        
        # Test OpenAPI JSON
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        
        schema = openapi_response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Real Estate Scraper"

    def test_api_response_models_match_actual_responses(self, client, sample_property_data, sample_prediction_data):
        """Test that actual API responses match the defined Pydantic models"""
        
        with patch('app.scrape_menzili') as mock_scrape, \
             patch('app.predict_price') as mock_predict:
            
            mock_scrape.return_value = sample_property_data
            mock_predict.return_value = sample_prediction_data
            
            response = client.post("/api/scrape", json={
                "provider": "menzili",
                "url": "https://www.menzili.tn/test"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            # Check ScrapeResponse model structure
            required_fields = ["success", "data", "prediction", "error"]
            for field in required_fields:
                assert field in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
