
import joblib
import pandas as pd
import numpy as np

def load_model():
    """Load the trained model and scaler."""
    model = joblib.load("best_model_20250923_115748.pkl")
    scaler = None
    return model, scaler

def predict_price(features_dict):
    """
    Predict house price from feature dictionary.
    
    Args:
        features_dict: Dictionary with feature names as keys and values as values
        
    Returns:
        Predicted price
    """
    model, scaler = load_model()
    
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Ensure all features are present
    expected_features = ['living_area', 'land_area', 'room_count', 'bathroom_count', 'price_per_sqm', 'total_area', 'living_to_land_ratio', 'avg_room_size', 'bathroom_room_ratio', 'description_length', 'description_word_count', 'avg_word_length', 'punctuation_density', 'luxury_score', 'ai_property_score', 'investment_potential', 'has_garage', 'has_garden', 'has_pool', 'has_balcony', 'has_terrace', 'has_elevator', 'has_security', 'is_appartement', 'is_maison', 'is_villa', 'is_terrain', 'is_apartment', 'is_house', 'is_local_commercial', 'is_duplex', 'is_studio', 'is_bureau', 'is_major_city', 'is_coastal', 'gov_tunis', 'gov_cap_bon', 'gov_nabeul', 'gov_bizerte', 'gov_mã©denine', 'gov_ariana', 'gov_unknown', 'gov_ben_arous', 'has_swimming_pool', 'has_modern', 'has_luxury', 'has_sea_view', 'has_furnished', 'is_excellent_condition', 'is_good_condition', 'is_needs_renovation', 'is_new_construction', 'property_tier_Basic', 'property_tier_Economy', 'property_tier_Premium', 'property_tier_Standard', 'market_segment_Budget', 'market_segment_Entry-Level', 'market_segment_Luxury', 'market_segment_Mid-Market', 'market_segment_Upper-Mid']
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing features
    
    # Reorder columns to match training
    df = df[expected_features]
    
    # Scale if necessary
    X = df.values
    
    # Predict
    prediction = model.predict(X)[0]
    
    return prediction

# Example usage:
# price = predict_price({"area": 100, "rooms": 3, "bathrooms": 2, ...})
