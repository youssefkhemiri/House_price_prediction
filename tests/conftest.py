# Pytest configuration for Real Estate Scraper API tests

import os
import sys
import pytest
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_property_data():
    """Fixture providing sample property data for tests"""
    return {
        "url": "https://www.menzili.tn/test-property",
        "id": "test-123",
        "source": "menzili",
        "property_type": "Maison",
        "price": "250000",
        "currency": "DT",
        "description": ["Beautiful house with garden and garage"],
        "address": "15 Rue de la République, Tunis",
        "governorate": "Tunis",
        "delegation": "Tunis Médina",
        "locality": "Centre Ville",
        "postal_code": "1001",
        "living_area": "150",
        "land_area": 300,
        "room_count": 4,
        "bathroom_count": 2,
        "construction_year": 2015,
        "floor": 0,
        "has_garage": True,
        "has_garden": True,
        "has_pool": False,
        "has_balcony": True,
        "has_terrace": True,
        "heating": False,
        "air_conditioning": True,
        "furnished": False,
        "phone": ["71123456", "98765432"],
        "agency": "Test Real Estate Agency",
        "contact_name": "Ahmed Ben Ali",
        "listing_date": 1640995200000,  # Timestamp
        "last_updated": None,
        "photos": [
            "https://example.com/photo1.jpg",
            "https://example.com/photo2.jpg"
        ],
        "features": ["garage", "garden", "balcony", "terrace", "air_conditioning"],
        "condition": "Good",
        "transaction_type": "Vente"
    }

@pytest.fixture
def sample_prediction_data():
    """Fixture providing sample prediction data for tests"""
    return {
        "predicted_price": 275000,
        "confidence": 0.85,
        "price_range": {
            "min": 250000,
            "max": 300000
        },
        "factors": [
            "location",
            "property_size",
            "property_type",
            "amenities",
            "market_trends"
        ],
        "model_version": "1.0",
        "prediction_date": "2025-09-01"
    }

@pytest.fixture
def sample_api_responses():
    """Fixture providing sample API response structures"""
    return {
        "success_response": {
            "success": True,
            "data": {
                "property_type": "Maison",
                "price": "250000",
                "currency": "DT"
            },
            "prediction": {
                "predicted_price": 275000,
                "confidence": 0.85
            },
            "error": None
        },
        "error_response": {
            "success": False,
            "data": None,
            "prediction": None,
            "error": "Failed to scrape property data"
        }
    }

@pytest.fixture
def mock_html_responses():
    """Fixture providing mock HTML responses for different sites"""
    return {
        "menzili_success": """
        <!DOCTYPE html>
        <html>
        <head><title>Villa à Tunis - Menzili</title></head>
        <body>
            <div class="annonce-title">
                <h1>Belle Villa avec Piscine - Sidi Bou Said</h1>
            </div>
            <div class="annonce-prix">
                <span>350,000 DT</span>
            </div>
            <div class="annonce-addr">
                <span>Sidi Bou Said, Tunis, Tunisie</span>
            </div>
            <div class="block-detail">
                <span>Chambres :</span><strong>5</strong>
                <span>Salle de bain :</span><strong>3</strong>
                <span>Piéces (Totale) :</span><strong>8</strong>
                <span>Surf habitable :</span><strong>200</strong>
                <span>Surf terrain :</span><strong>400</strong>
            </div>
            <div class="span-opts">
                <strong>Garage</strong>
                <strong>Jardin</strong>
                <strong>Piscine</strong>
                <strong>Terrasse</strong>
                <strong>Climatisation</strong>
            </div>
            <div class="annonce-description">
                <p>Magnifique villa située dans le prestigieux quartier de Sidi Bou Said.</p>
                <p>Cette propriété exceptionnelle offre un cadre de vie idéal.</p>
            </div>
            <div class="annonce-photos">
                <img src="https://menzili.tn/photo1.jpg" alt="Photo 1">
                <img src="https://menzili.tn/photo2.jpg" alt="Photo 2">
            </div>
        </body>
        </html>
        """,
        
        "mubawab_success": """
        <!DOCTYPE html>
        <html>
        <head><title>Appartement à Tunis - Mubawab</title></head>
        <body>
            <h1 class="searchResultsPropertyTitle">Appartement Moderne - Centre Ville Tunis</h1>
            <div class="searchResultsPropertyPrice">
                <span>180,000 DT</span>
            </div>
            <div class="searchResultsPropertyLocation">
                <span>Centre Ville, Tunis</span>
            </div>
            <div class="searchResultsPropertyDetails">
                <div>3 chambres</div>
                <div>2 salles de bain</div>
                <div>120 m²</div>
                <div>2ème étage</div>
            </div>
            <div class="searchResultsPropertyDescription">
                <p>Appartement moderne avec finitions de qualité.</p>
                <p>Situé dans un quartier résidentiel calme.</p>
            </div>
        </body>
        </html>
        """,
        
        "empty_page": """
        <!DOCTYPE html>
        <html>
        <head><title>Page Not Found</title></head>
        <body>
            <h1>404 - Page Not Found</h1>
        </body>
        </html>
        """
    }

@pytest.fixture
def test_urls():
    """Fixture providing test URLs for different providers"""
    return {
        "menzili": [
            "https://www.menzili.tn/annonce/villa-sidi-bou-said-123",
            "https://www.menzili.tn/annonce/appartement-tunis-456"
        ],
        "mubawab": [
            "https://www.mubawab.tn/fr/ct/tunis/vente-appartement-123",
            "https://www.mubawab.tn/fr/ct/ariana/location-maison-456"
        ],
        "any_website": [
            "https://example-realestate.com/property/123",
            "https://another-site.com/house-for-sale/456"
        ]
    }

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup code would go here if needed
    pass

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )

# Pytest collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark network tests
        if any(keyword in item.nodeid.lower() for keyword in ["network", "http", "requests"]):
            item.add_marker(pytest.mark.network)
        
        # Mark slow tests
        if any(keyword in item.nodeid.lower() for keyword in ["slow", "timeout"]):
            item.add_marker(pytest.mark.slow)
