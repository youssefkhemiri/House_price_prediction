#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Model Demo Script for House Price Prediction

This script provides an easy interface to test the trained models
without command line arguments.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_models import ModelTester, load_latest_model
import json


def demo_predictions():
    """Demonstrate model predictions with sample houses."""
    print("🏠 House Price Prediction Model Demo")
    print("=" * 50)
    
    try:
        # Load the latest model
        metadata_path = load_latest_model()
        tester = ModelTester(metadata_path)
        
        # Print basic info
        print(f"📅 Model trained: {tester.metadata['timestamp']}")
        print(f"🤖 Available models: {list(tester.models.keys())}")
        print()
        
        # Sample houses to test
        test_houses = [
            {
                "name": "Small Apartment",
                "data": {
                    "room_count": 2,
                    "bathroom_count": 1,
                    "land_area": 80,
                    "has_garage": 0,
                    "has_garden": 0,
                    "has_pool": 0,
                    "has_balcony": 1,
                    "has_terrace": 0,
                    "heating": 1,
                    "air_conditioning": 0,
                    "quality_score": 3,
                    "price": 200000  # Required for preprocessing
                }
            },
            {
                "name": "Family House",
                "data": {
                    "room_count": 4,
                    "bathroom_count": 2,
                    "land_area": 200,
                    "has_garage": 1,
                    "has_garden": 1,
                    "has_pool": 0,
                    "has_balcony": 1,
                    "has_terrace": 1,
                    "heating": 1,
                    "air_conditioning": 1,
                    "quality_score": 4,
                    "price": 400000  # Required for preprocessing
                }
            },
            {
                "name": "Luxury Villa",
                "data": {
                    "room_count": 6,
                    "bathroom_count": 3,
                    "land_area": 500,
                    "has_garage": 1,
                    "has_garden": 1,
                    "has_pool": 1,
                    "has_balcony": 1,
                    "has_terrace": 1,
                    "heating": 1,
                    "air_conditioning": 1,
                    "quality_score": 5,
                    "price": 800000  # Required for preprocessing
                }
            }
        ]
        
        # Make predictions for each house
        for house in test_houses:
            print(f"🏡 {house['name']}")
            print("-" * 30)
            
            # Show input features
            features = house['data'].copy()
            features.pop('price')  # Don't show the dummy price
            
            print("Features:")
            for key, value in features.items():
                print(f"   {key}: {value}")
            
            # Get predictions
            predictions = tester.predict_single(house['data'])
            
            print("\nPrice Predictions:")
            for model_name, pred in predictions.items():
                print(f"   {model_name:15}: {pred['price']:8,.0f} TND")
            
            print("\n" + "="*50 + "\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_prediction():
    """Interactive prediction mode."""
    print("\n🎯 Interactive House Price Prediction")
    print("=" * 40)
    
    try:
        # Load model
        metadata_path = load_latest_model()
        tester = ModelTester(metadata_path)
        
        print("Enter house details (press Enter for default values):")
        
        # Get user input
        house_data = {}
        
        # Numeric features
        numeric_features = [
            ('room_count', 'Number of rooms', 3),
            ('bathroom_count', 'Number of bathrooms', 2),
            ('land_area', 'Land area (m²)', 150),
            ('quality_score', 'Quality score (1-5)', 3)
        ]
        
        for feature, description, default in numeric_features:
            while True:
                try:
                    user_input = input(f"{description} (default {default}): ").strip()
                    if user_input == "":
                        house_data[feature] = default
                        break
                    else:
                        house_data[feature] = float(user_input)
                        break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Boolean features
        boolean_features = [
            ('has_garage', 'Has garage'),
            ('has_garden', 'Has garden'),
            ('has_pool', 'Has pool'),
            ('has_balcony', 'Has balcony'),
            ('has_terrace', 'Has terrace'),
            ('heating', 'Has heating'),
            ('air_conditioning', 'Has air conditioning')
        ]
        
        for feature, description in boolean_features:
            while True:
                user_input = input(f"{description} (y/n, default n): ").strip().lower()
                if user_input in ['', 'n', 'no']:
                    house_data[feature] = 0
                    break
                elif user_input in ['y', 'yes']:
                    house_data[feature] = 1
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
        
        # Add dummy price for preprocessing
        house_data['price'] = 300000
        
        print("\n🔮 Making predictions...")
        predictions = tester.predict_single(house_data)
        
        print(f"\n🏠 Price Predictions for Your House:")
        print("-" * 40)
        for model_name, pred in predictions.items():
            print(f"{model_name:15}: {pred['price']:8,.0f} TND")
        
        # Show recommended price (ensemble average)
        if 'ensemble' in predictions:
            recommended = predictions['ensemble']['price']
            print(f"\n💡 Recommended price: {recommended:,.0f} TND")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*60)
        print("🏠 HOUSE PRICE PREDICTION SYSTEM")
        print("="*60)
        print("1. View model information")
        print("2. Demo predictions (sample houses)")
        print("3. Interactive prediction (enter your house details)")
        print("4. Exit")
        print("-" * 60)
        
        choice = input("Choose an option (1-4): ").strip()
        
        if choice == "1":
            try:
                metadata_path = load_latest_model()
                tester = ModelTester(metadata_path)
                tester.print_model_info()
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            demo_predictions()
        
        elif choice == "3":
            interactive_prediction()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
