#!/usr/bin/env python3
"""
Pytest tests for the scraper modules
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scrapers.menzili_scraper import scrape_menzili
from scrapers.mubawab_scraper import scrape_mubawab
from scrapers.any_website_scraper import scrape_any_website

class TestMenziliScraper:
    """Test cases for Menzili scraper"""

    @patch('scrapers.menzili_scraper.requests.get')
    def test_scrape_menzili_success(self, mock_get):
        """Test successful Menzili scraping"""
        # Mock HTML response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Property</title></head>
            <body>
                <div class="annonce-title">
                    <h1>Beautiful Villa in Tunis</h1>
                </div>
                <div class="annonce-prix">
                    <span>250,000 DT</span>
                </div>
                <div class="annonce-addr">
                    <span>Tunis, La Marsa</span>
                </div>
                <div class="block-detail">
                    <span>Chambres :</span><strong>4</strong>
                    <span>Salle de bain :</span><strong>2</strong>
                    <span>Surf habitable :</span><strong>150</strong>
                </div>
                <div class="span-opts">
                    <strong>Garage</strong>
                    <strong>Jardin</strong>
                </div>
                <div class="annonce-description">
                    <p>Beautiful property with garden and garage.</p>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = scrape_menzili("https://www.menzili.tn/test-property")
        
        assert isinstance(result, dict)
        assert result["source"] == "menzili"
        assert "url" in result
        assert "property_type" in result
        assert "price" in result

    @patch('scrapers.menzili_scraper.requests.get')
    def test_scrape_menzili_network_error(self, mock_get):
        """Test Menzili scraper with network error"""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            scrape_menzili("https://www.menzili.tn/test-property")

    @patch('scrapers.menzili_scraper.requests.get')
    def test_scrape_menzili_empty_response(self, mock_get):
        """Test Menzili scraper with empty response"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_get.return_value = mock_response
        
        result = scrape_menzili("https://www.menzili.tn/test-property")
        
        assert isinstance(result, dict)
        assert result["source"] == "menzili"

    def test_scrape_menzili_invalid_url(self):
        """Test Menzili scraper with invalid URL"""
        with pytest.raises(Exception):
            scrape_menzili("not-a-url")

class TestMubawabScraper:
    """Test cases for Mubawab scraper"""

    @patch('scrapers.mubawab_scraper.requests.get')
    def test_scrape_mubawab_success(self, mock_get):
        """Test successful Mubawab scraping"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Property - Mubawab</title></head>
            <body>
                <h1 class="searchResultsPropertyTitle">Apartment in Tunis</h1>
                <div class="searchResultsPropertyPrice">
                    <span>300,000 DT</span>
                </div>
                <div class="searchResultsPropertyLocation">
                    <span>Tunis, Carthage</span>
                </div>
                <div class="searchResultsPropertyDetails">
                    <span>3 chambres</span>
                    <span>2 salles de bain</span>
                    <span>120 m²</span>
                </div>
                <div class="searchResultsPropertyDescription">
                    <p>Modern apartment with sea view.</p>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        result = scrape_mubawab("https://www.mubawab.tn/test-property")
        
        assert isinstance(result, dict)
        assert result["source"] == "mubawab"
        assert "url" in result
        assert "property_type" in result

    @patch('scrapers.mubawab_scraper.requests.get')
    def test_scrape_mubawab_network_error(self, mock_get):
        """Test Mubawab scraper with network error"""
        mock_get.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            scrape_mubawab("https://www.mubawab.tn/test-property")

class TestAnyWebsiteScraper:
    """Test cases for Any Website scraper"""

    def test_scrape_any_website_returns_mock_data(self):
        """Test that Any Website scraper returns mock data"""
        result = scrape_any_website("https://example.com/property")
        
        assert isinstance(result, dict)
        assert result["source"] == "any_website"
        assert result["url"] == "https://example.com/property"
        assert "property_type" in result
        assert "price" in result
        assert "description" in result

    def test_scrape_any_website_different_urls(self):
        """Test Any Website scraper with different URLs"""
        urls = [
            "https://example1.com/property",
            "https://example2.com/house",
            "https://test.com/apartment"
        ]
        
        for url in urls:
            result = scrape_any_website(url)
            assert result["url"] == url
            assert result["source"] == "any_website"

class TestScraperUtilities:
    """Test utility functions used by scrapers"""

    def test_phone_regex_pattern(self):
        """Test phone number extraction patterns"""
        # This would test any utility functions if they were exposed
        # For now, we'll test that scrapers handle phone extraction
        pass

    def test_price_parsing(self):
        """Test price parsing logic"""
        # Test that scrapers can handle various price formats
        test_cases = [
            "250,000 DT",
            "250.000 DT",
            "250000 DT",
            "DT 250,000",
        ]
        # This would be implemented if price parsing was exposed as a utility
        pass

    def test_address_extraction(self):
        """Test address extraction logic"""
        # Test address parsing from various formats
        pass

class TestScraperDataStructure:
    """Test that all scrapers return consistent data structure"""

    def test_menzili_data_structure(self):
        """Test Menzili scraper returns expected fields"""
        expected_fields = {
            "url", "id", "source", "property_type", "price", "currency",
            "description", "address", "governorate", "delegation", "locality",
            "living_area", "room_count", "bathroom_count", "has_garage",
            "has_garden", "has_pool", "photos", "features", "transaction_type"
        }
        
        with patch('scrapers.menzili_scraper.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body><h1>Test</h1></body></html>"
            mock_get.return_value = mock_response
            
            result = scrape_menzili("https://www.menzili.tn/test")
            
            # Check that most expected fields are present
            common_fields = expected_fields.intersection(set(result.keys()))
            assert len(common_fields) > 10  # Should have most fields

    def test_any_website_data_structure(self):
        """Test Any Website scraper returns expected fields"""
        result = scrape_any_website("https://example.com/property")
        
        expected_fields = {
            "url", "source", "property_type", "price", "currency",
            "description", "address", "room_count", "bathroom_count"
        }
        
        for field in expected_fields:
            assert field in result

    def test_all_scrapers_have_required_fields(self):
        """Test that all scrapers return required fields"""
        required_fields = ["url", "source", "property_type"]
        
        # Test any_website scraper (doesn't require network)
        result = scrape_any_website("https://example.com/test")
        for field in required_fields:
            assert field in result
            
        # Mock test for other scrapers
        with patch('scrapers.menzili_scraper.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body></body></html>"
            mock_get.return_value = mock_response
            
            result = scrape_menzili("https://www.menzili.tn/test")
            for field in required_fields:
                assert field in result

class TestScraperErrorHandling:
    """Test error handling in scrapers"""

    @patch('scrapers.menzili_scraper.requests.get')
    def test_menzili_http_error(self, mock_get):
        """Test Menzili scraper with HTTP error"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            scrape_menzili("https://www.menzili.tn/nonexistent")

    @patch('scrapers.mubawab_scraper.requests.get')
    def test_mubawab_timeout(self, mock_get):
        """Test Mubawab scraper with timeout"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(Exception):
            scrape_mubawab("https://www.mubawab.tn/test")

    def test_any_website_never_fails(self):
        """Test that Any Website scraper never raises exceptions"""
        test_urls = [
            "",
            "invalid-url",
            None,
            "https://nonexistent-domain-12345.com"
        ]
        
        for url in test_urls:
            try:
                result = scrape_any_website(str(url) if url is not None else "")
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Any Website scraper should not raise exceptions, but got: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
