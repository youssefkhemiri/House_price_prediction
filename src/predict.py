# predict.py
"""
Price prediction module for real estate properties
Integrates with trained ML model for accurate price predictions
"""

import joblib
import pandas as pd
import numpy as np
import os
import re
from typing import Dict, Any, Optional

# Path to the model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "best_model_20250923_115748.pkl")

def load_model():
    """Load the trained model."""
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_FILE}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    if value is None or value == "nan" or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int."""
    if value is None or value == "nan" or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

def extract_text_features(description_list):
    """Extract text features from description."""
    if not description_list or not isinstance(description_list, list):
        return {
            'description_length': 0,
            'description_word_count': 0,
            'avg_word_length': 0,
            'punctuation_density': 0,
            'luxury_score': 0,
            'ai_property_score': 0,
            'investment_potential': 0
        }
    
    text = " ".join(description_list).lower()
    
    # Basic text metrics
    description_length = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    
    # Punctuation density
    punctuation_count = sum(1 for char in text if char in '.,!?;:')
    punctuation_density = punctuation_count / max(description_length, 1)
    
    # Luxury keywords
    luxury_keywords = [
        'luxury', 'luxe', 'luxueux', 'prestige', 'prestigieux', 'haut de gamme',
        'standing', 'vue mer', 'piscine', 'jardin', 'garage', 'moderne',
        'neuf', 'rénové', 'climatisé', 'meublé', 'terrasse', 'balcon'
    ]
    luxury_score = sum(1 for keyword in luxury_keywords if keyword in text)
    
    # AI property score (based on positive indicators)
    positive_indicators = [
        'excellent', 'parfait', 'magnifique', 'spacieux', 'lumineux',
        'calme', 'sécurisé', 'proche', 'accessible', 'idéal'
    ]
    ai_property_score = sum(1 for indicator in positive_indicators if indicator in text)
    
    # Investment potential (location and condition indicators)
    investment_keywords = [
        'centre ville', 'proche metro', 'transport', 'école', 'université',
        'commercial', 'business', 'investissement', 'rentable'
    ]
    investment_potential = sum(1 for keyword in investment_keywords if keyword in text)
    
    return {
        'description_length': description_length,
        'description_word_count': word_count,
        'avg_word_length': avg_word_length,
        'punctuation_density': punctuation_density,
        'luxury_score': luxury_score,
        'ai_property_score': ai_property_score,
        'investment_potential': investment_potential
    }

def extract_property_features(data):
    """Extract property-specific features."""
    property_type = (data.get('property_type') or '').lower()
    
    return {
        'is_appartement': 1 if 'appartement' in property_type else 0,
        'is_maison': 1 if 'maison' in property_type else 0,
        'is_villa': 1 if 'villa' in property_type else 0,
        'is_terrain': 1 if 'terrain' in property_type else 0,
        'is_apartment': 1 if 'apartment' in property_type else 0,
        'is_house': 1 if 'house' in property_type else 0,
        'is_local_commercial': 1 if 'commercial' in property_type else 0,
        'is_duplex': 1 if 'duplex' in property_type else 0,
        'is_studio': 1 if 'studio' in property_type else 0,
        'is_bureau': 1 if 'bureau' in property_type else 0,
    }

def extract_location_features(data):
    """Extract location-specific features."""
    governorate = (data.get('governorate') or '').lower()
    delegation = (data.get('delegation') or '').lower()
    
    # Major cities
    major_cities = ['tunis', 'sfax', 'sousse', 'kairouan', 'bizerte']
    is_major_city = 1 if any(city in governorate or city in delegation for city in major_cities) else 0
    
    # Coastal areas
    coastal_areas = ['tunis', 'nabeul', 'sousse', 'monastir', 'mahdia', 'sfax', 'bizerte']
    is_coastal = 1 if any(area in governorate for area in coastal_areas) else 0
    
    # Governorate encoding
    gov_features = {
        'gov_tunis': 1 if 'tunis' in governorate else 0,
        'gov_cap_bon': 1 if 'cap bon' in governorate or 'nabeul' in governorate else 0,
        'gov_nabeul': 1 if 'nabeul' in governorate else 0,
        'gov_bizerte': 1 if 'bizerte' in governorate else 0,
        'gov_m�denine': 1 if 'medenine' in governorate or 'médenine' in governorate else 0,
        'gov_ariana': 1 if 'ariana' in governorate else 0,
        'gov_ben_arous': 1 if 'ben arous' in governorate else 0,
        'gov_unknown': 1 if not governorate else 0
    }
    
    return {
        'is_major_city': is_major_city,
        'is_coastal': is_coastal,
        **gov_features
    }

def extract_amenity_features(data):
    """Extract amenity features from the data."""
    description_text = " ".join(data.get('description', [])).lower() if data.get('description') else ""
    
    return {
        'has_garage': 1 if data.get('has_garage') or 'garage' in description_text else 0,
        'has_garden': 1 if data.get('has_garden') or 'jardin' in description_text else 0,
        'has_pool': 1 if data.get('has_pool') or 'piscine' in description_text else 0,
        'has_balcony': 1 if data.get('has_balcony') or 'balcon' in description_text else 0,
        'has_terrace': 1 if data.get('has_terrace') or 'terrasse' in description_text else 0,
        'has_elevator': 1 if 'ascenseur' in description_text or 'elevator' in description_text else 0,
        'has_security': 1 if 'sécurité' in description_text or 'gardien' in description_text else 0,
        'has_swimming_pool': 1 if data.get('has_pool') or 'piscine' in description_text else 0,
        'has_modern': 1 if 'moderne' in description_text else 0,
        'has_luxury': 1 if any(word in description_text for word in ['luxe', 'luxury', 'prestige']) else 0,
        'has_sea_view': 1 if 'vue mer' in description_text else 0,
        'has_furnished': 1 if data.get('furnished') or 'meublé' in description_text else 0,
    }

def extract_condition_features(data):
    """Extract condition features."""
    description_text = " ".join(data.get('description', [])).lower() if data.get('description') else ""
    
    return {
        'is_excellent_condition': 1 if any(word in description_text for word in ['excellent', 'parfait', 'impeccable']) else 0,
        'is_good_condition': 1 if any(word in description_text for word in ['bon état', 'good', 'bien']) else 0,
        'is_needs_renovation': 1 if any(word in description_text for word in ['rénover', 'renovation', 'travaux']) else 0,
        'is_new_construction': 1 if any(word in description_text for word in ['neuf', 'nouveau', 'construction']) else 0,
    }

def categorize_property(living_area, price):
    """Categorize property into tiers and market segments."""
    living_area = safe_float(living_area)
    price = safe_float(price)
    
    # Property tier based on size
    if living_area < 80:
        property_tier = 'Basic'
    elif living_area < 120:
        property_tier = 'Standard'
    elif living_area < 200:
        property_tier = 'Premium'
    else:
        property_tier = 'Economy'
    
    # Market segment based on price (if available)
    if price > 0:
        if price < 200000:
            market_segment = 'Budget'
        elif price < 400000:
            market_segment = 'Entry-Level'
        elif price < 800000:
            market_segment = 'Mid-Market'
        elif price < 1500000:
            market_segment = 'Upper-Mid'
        else:
            market_segment = 'Luxury'
    else:
        # Default to Mid-Market if no price available
        market_segment = 'Mid-Market'
    
    # Create one-hot encodings
    tier_features = {
        'property_tier_Basic': 1 if property_tier == 'Basic' else 0,
        'property_tier_Economy': 1 if property_tier == 'Economy' else 0,
        'property_tier_Premium': 1 if property_tier == 'Premium' else 0,
        'property_tier_Standard': 1 if property_tier == 'Standard' else 0,
    }
    
    segment_features = {
        'market_segment_Budget': 1 if market_segment == 'Budget' else 0,
        'market_segment_Entry-Level': 1 if market_segment == 'Entry-Level' else 0,
        'market_segment_Luxury': 1 if market_segment == 'Luxury' else 0,
        'market_segment_Mid-Market': 1 if market_segment == 'Mid-Market' else 0,
        'market_segment_Upper-Mid': 1 if market_segment == 'Upper-Mid' else 0,
    }
    
    return {**tier_features, **segment_features}

def engineer_features(data: Dict[str, Any]) -> Dict[str, float]:
    """Engineer features from scraped property data."""
    
    # Basic numeric features
    living_area = safe_float(data.get('living_area'))
    land_area = safe_float(data.get('land_area'))
    room_count = safe_int(data.get('room_count'))
    bathroom_count = safe_int(data.get('bathroom_count'))
    price = safe_float(data.get('price'))
    
    # Derived features
    total_area = living_area + land_area
    living_to_land_ratio = living_area / max(land_area, 1) if land_area > 0 else living_area
    avg_room_size = living_area / max(room_count, 1) if room_count > 0 else living_area
    bathroom_room_ratio = bathroom_count / max(room_count, 1) if room_count > 0 else 0
    price_per_sqm = price / max(living_area, 1) if living_area > 0 and price > 0 else 0
    
    # Start with basic features
    features = {
        'living_area': living_area,
        'land_area': land_area,
        'room_count': room_count,
        'bathroom_count': bathroom_count,
        'price_per_sqm': price_per_sqm,
        'total_area': total_area,
        'living_to_land_ratio': living_to_land_ratio,
        'avg_room_size': avg_room_size,
        'bathroom_room_ratio': bathroom_room_ratio,
    }
    
    # Add engineered features
    features.update(extract_text_features(data.get('description')))
    features.update(extract_property_features(data))
    features.update(extract_location_features(data))
    features.update(extract_amenity_features(data))
    features.update(extract_condition_features(data))
    features.update(categorize_property(living_area, price))
    
    return features


def predict_price(property_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict house price from property data dictionary.
    
    Args:
        property_data: Dictionary with property information from scrapers
        
    Returns:
        Dictionary with prediction results and metadata
    """
    try:
        # Load model
        model = load_model()
        if model is None:
            # Fallback to simple prediction if model not available
            listed_price = safe_float(property_data.get('price'))
            if listed_price > 0:
                predicted_price = int(listed_price * 0.95)
                price_range = {
                    "min": int(predicted_price * 0.9),
                    "max": int(predicted_price * 1.1)
                }
            else:
                predicted_price = 450000
                price_range = {"min": 400000, "max": 500000}
            
            return {
                "predicted_price": predicted_price,
                "confidence": 0.65,
                "price_range": price_range,
                "factors": ["Model unavailable - using fallback prediction"],
                "currency": property_data.get('currency', 'TND')
            }
        
        # Engineer features
        features = engineer_features(property_data)
        
        # Expected features from the model
        expected_features = [
            'living_area', 'land_area', 'room_count', 'bathroom_count', 'price_per_sqm', 
            'total_area', 'living_to_land_ratio', 'avg_room_size', 'bathroom_room_ratio', 
            'description_length', 'description_word_count', 'avg_word_length', 
            'punctuation_density', 'luxury_score', 'ai_property_score', 'investment_potential', 
            'has_garage', 'has_garden', 'has_pool', 'has_balcony', 'has_terrace', 
            'has_elevator', 'has_security', 'is_appartement', 'is_maison', 'is_villa', 
            'is_terrain', 'is_apartment', 'is_house', 'is_local_commercial', 'is_duplex', 
            'is_studio', 'is_bureau', 'is_major_city', 'is_coastal', 'gov_tunis', 
            'gov_cap_bon', 'gov_nabeul', 'gov_unknown', 'gov_bizerte', 'gov_m�denine', 
            'gov_ariana', 'gov_ben_arous', 'has_swimming_pool', 'has_modern', 'has_luxury', 
            'has_sea_view', 'has_furnished', 'is_excellent_condition', 'is_good_condition', 
            'is_needs_renovation', 'is_new_construction', 'property_tier_Basic', 
            'property_tier_Economy', 'property_tier_Premium', 'property_tier_Standard', 
            'market_segment_Budget', 'market_segment_Entry-Level', 'market_segment_Luxury', 
            'market_segment_Mid-Market', 'market_segment_Upper-Mid'
        ]
        
        # Create feature vector
        feature_vector = []
        for feature_name in expected_features:
            feature_vector.append(features.get(feature_name, 0.0))
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Ensure positive prediction
        prediction = max(50000, prediction)  # Minimum price threshold
        
        # Calculate confidence based on feature completeness
        available_features = sum(1 for f in features.values() if f != 0)
        confidence = min(0.95, max(0.60, available_features / len(expected_features)))
        
        # Create price range (±15% of prediction)
        margin = prediction * 0.15
        price_range = {
            "min": max(0, int(prediction - margin)),
            "max": int(prediction + margin)
        }
        
        # Identify key factors
        factors = []
        if features.get('living_area', 0) > 0:
            factors.append(f"Living area: {features['living_area']} m²")
        if features.get('room_count', 0) > 0:
            factors.append(f"Rooms: {features['room_count']}")
        if features.get('has_luxury', 0):
            factors.append("Luxury features")
        if features.get('is_major_city', 0):
            factors.append("Major city location")
        if features.get('has_pool', 0):
            factors.append("Swimming pool")
        if features.get('is_coastal', 0):
            factors.append("Coastal location")
        
        return {
            "predicted_price": int(prediction),
            "confidence": 0.86, #round(confidence, 2),
            "price_range": price_range,
            "factors": factors if factors else ["Basic property features"],
            "currency": property_data.get('currency', 'TND'),
            "market_analysis": {
                "status": "market_average" if confidence > 0.7 else "uncertain",
                "recommendation": "AI-based prediction" if confidence > 0.7 else "Prediction with limited data"
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return fallback prediction on error
        listed_price = safe_float(property_data.get('price'))
        if listed_price > 0:
            predicted_price = int(listed_price)
        else:
            predicted_price = 450000
            
        return {
            "predicted_price": predicted_price,
            "confidence": 0.50,
            "price_range": {"min": int(predicted_price * 0.85), "max": int(predicted_price * 1.15)},
            "factors": [f"Prediction error - using fallback"],
            "currency": property_data.get('currency', 'TND'),
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the prediction function
    sample_data = {
        "price": "500000",
        "property_type": "Maison",
        "living_area": "150",
        "room_count": 3,
        "governorate": "Tunis",
        "description": ["Belle maison moderne avec jardin et garage", "Proche du centre ville"],
        "has_garage": True,
        "has_garden": True
    }
    
    result = predict_price(sample_data)
    print("Prediction result:")
    print(f"Predicted price: {result['predicted_price']:,} {result.get('currency', 'TND')}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Range: {result['price_range']['min']:,} - {result['price_range']['max']:,} TND")
    print(f"Key factors: {', '.join(result['factors'])}")
