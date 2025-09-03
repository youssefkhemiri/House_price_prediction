#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Testing Script for House Price Prediction

This script loads saved models and tests them on new data or validation sets.
It provides functionality to:
- Load saved models and preprocessors
- Test models on new data
- Compare model performance
- Make predictions on individual samples
- Validate model consistency

Usage:
    python test_models.py --model_dir models --data_path path/to/test_data.json
"""

import pandas as pd
import json
import os
import argparse
import joblib
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class ModelTester:
    """Class for loading and testing saved house price prediction models."""
    
    def __init__(self, model_metadata_path):
        """
        Initialize model tester with metadata file.
        
        Args:
            model_metadata_path: Path to model metadata JSON file
        """
        self.metadata_path = model_metadata_path
        self.metadata = self._load_metadata()
        self.models = {}
        self.preprocessors = {}
        self.feature_names = self.metadata['feature_names']
        
        # Load models and preprocessors
        self._load_models()
        self._load_preprocessors()
    
    def _load_metadata(self):
        """Load model metadata from JSON file."""
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✅ Loaded metadata from: {self.metadata_path}")
            return metadata
        except Exception as e:
            raise Exception(f"Failed to load metadata: {e}")
    
    def _load_models(self):
        """Load all saved models."""
        print("\n📥 Loading trained models...")
        model_paths = self.metadata.get('model_paths', {})
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    print(f"✅ Loaded {model_name} from: {model_path}")
                else:
                    print(f"❌ Model file not found: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    def _load_preprocessors(self):
        """Load all saved preprocessors."""
        print("\n📥 Loading preprocessors...")
        preprocessor_paths = self.metadata.get('preprocessor_paths', {})
        
        for prep_name, prep_path in preprocessor_paths.items():
            try:
                if os.path.exists(prep_path):
                    preprocessor = joblib.load(prep_path)
                    self.preprocessors[prep_name] = preprocessor
                    print(f"✅ Loaded {prep_name} from: {prep_path}")
                else:
                    print(f"⚠️  Preprocessor file not found: {prep_path}")
            except Exception as e:
                print(f"❌ Failed to load {prep_name}: {e}")
    
    def print_model_info(self):
        """Print information about loaded models."""
        print("\n" + "="*60)
        print("📊 MODEL INFORMATION")
        print("="*60)
        
        training_metadata = self.metadata.get('training_metadata', {})
        print(f"📅 Training Date: {self.metadata.get('timestamp', 'Unknown')}")
        print(f"📁 Data Source: {training_metadata.get('data_source', 'Unknown')}")
        print(f"🔤 TF-IDF Used: {training_metadata.get('use_tfidf', False)}")
        print(f"📏 Data Shape: {training_metadata.get('data_shape', 'Unknown')}")
        print(f"🎯 Target Transform: {training_metadata.get('target_transform', 'Unknown')}")
        print(f"🏋️  Training Samples: {training_metadata.get('training_samples', 'Unknown')}")
        print(f"🧪 Test Samples: {training_metadata.get('test_samples', 'Unknown')}")
        
        print(f"\n🤖 Loaded Models: {list(self.models.keys())}")
        print(f"🔧 Loaded Preprocessors: {list(self.preprocessors.keys())}")
        print(f"📋 Feature Count: {len(self.feature_names)}")
        
        # Show training performance
        results = training_metadata.get('results', {})
        if results:
            print(f"\n📈 Training Performance:")
            for model_name, metrics in results.items():
                r2 = metrics.get('r2_score', 0)
                mae = metrics.get('mae', 0)
                print(f"   {model_name}: R² = {r2:.3f}, MAE = {mae:.2f}")
    
    def preprocess_data(self, df):
        """
        Preprocess new data using the same pipeline as training.
        
        Args:
            df: DataFrame with raw house data
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        df = df.copy()
        
        # Apply the same preprocessing as in training
        df = df[df['price'].notnull() & (df['price'] > 1000)]
        
        # Handle TF-IDF if it was used in training
        use_tfidf = self.metadata['training_metadata'].get('use_tfidf', False)
        if use_tfidf and 'description' in df.columns:
            df['description_str'] = df['description'].apply(self._flatten_to_str)
        
        # Drop columns that were dropped in training
        df = df.drop(columns=['address', 'governorate', 'locality', 'delegation', 'description'], errors='ignore')
        
        # Feature engineering
        if 'room_count' in df.columns and 'land_area' in df.columns:
            df['room_area'] = df['room_count'].fillna(0) * df['land_area'].fillna(0)
        
        # Cap outliers (using same quantiles as training)
        df = self._cap_outliers(df, 'price')
        if 'land_area' in df.columns:
            df = self._cap_outliers(df, 'land_area')
        
        # Log transforms
        df['log_price'] = np.log1p(df['price'])
        if 'land_area' in df.columns:
            df['log_land_area'] = np.log1p(df['land_area'])
        
        # Get features (excluding target and description_str for now)
        features = [col for col in df.columns if col not in ['price', 'log_price', 'description_str']]
        
        # Apply imputation
        imputer = SimpleImputer(strategy='median')
        df[features] = imputer.fit_transform(df[features])
        
        # Handle TF-IDF processing
        X = df[features]
        if use_tfidf and 'description_str' in df.columns and 'tfidf_vectorizer' in self.preprocessors:
            tfidf_vectorizer = self.preprocessors['tfidf_vectorizer']
            tfidf_matrix = tfidf_vectorizer.transform(df['description_str'])
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
                index=df.index
            )
            X = pd.concat([X, tfidf_df], axis=1)
        
        # Ensure we have the same features as training
        missing_features = set(self.feature_names) - set(X.columns)
        extra_features = set(X.columns) - set(self.feature_names)
        
        if missing_features:
            print(f"⚠️  Missing features (will be filled with 0): {missing_features}")
            for feature in missing_features:
                X[feature] = 0
        
        if extra_features:
            print(f"⚠️  Extra features (will be dropped): {extra_features}")
            X = X.drop(columns=extra_features)
        
        # Reorder columns to match training
        X = X[self.feature_names]
        
        return X, df['log_price']
    
    def _flatten_to_str(self, x):
        """Helper function to flatten lists to strings."""
        if isinstance(x, list):
            flat = []
            for item in x:
                if isinstance(item, list):
                    flat.extend(str(subitem) for subitem in item)
                else:
                    flat.append(str(item))
            return ' '.join(flat)
        return str(x)
    
    def _cap_outliers(self, df, col, lower_quantile=0.01, upper_quantile=0.99):
        """Cap outliers in a column."""
        if col in df.columns:
            lower = df[col].quantile(lower_quantile)
            upper = df[col].quantile(upper_quantile)
            df[col] = np.clip(df[col], lower, upper)
        return df
    
    def test_models(self, test_data_path):
        """
        Test all loaded models on test data.
        
        Args:
            test_data_path: Path to test data JSON file
        """
        print(f"\n🧪 Testing models on: {test_data_path}")
        
        # Load test data
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            df_test = pd.DataFrame(data)
            print(f"📊 Loaded {len(df_test)} test samples")
        except Exception as e:
            print(f"❌ Failed to load test data: {e}")
            return
        
        # Preprocess test data
        try:
            X_test, y_test = self.preprocess_data(df_test)
            print(f"✅ Preprocessed test data: {X_test.shape}")
        except Exception as e:
            print(f"❌ Failed to preprocess test data: {e}")
            return
        
        # Test each model
        print(f"\n📈 Model Performance on Test Data:")
        print("-" * 60)
        
        results = {}
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Make predictions
                y_pred_log = model.predict(X_test)
                y_pred = np.expm1(y_pred_log)  # Convert back from log
                y_true = np.expm1(y_test)
                
                # Calculate metrics
                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
                results[model_name] = {
                    'r2_score': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape
                }
                
                predictions[model_name] = {
                    'y_pred': y_pred,
                    'y_pred_log': y_pred_log
                }
                
                print(f"{model_name:20} | R² = {r2:6.3f} | MAE = {mae:8.2f} | RMSE = {rmse:8.2f} | MAPE = {mape:6.2f}%")
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
        
        # Show sample predictions
        print(f"\n🔍 Sample Predictions (First 10 samples):")
        print("-" * 100)
        print(f"{'Index':<8} | {'True Price':<12} | {'RF':<10} | {'XGB':<10} | {'LR':<10} | {'Ensemble':<10}")
        print("-" * 100)
        
        y_true = np.expm1(y_test)
        for i in range(min(10, len(y_true))):
            row = f"{i:<8} | {y_true.iloc[i]:12.2f} |"
            for model_name in ['random_forest', 'xgboost', 'linear_regression', 'ensemble']:
                if model_name in predictions:
                    pred = predictions[model_name]['y_pred'][i]
                    row += f" {pred:9.2f} |"
                else:
                    row += f" {'N/A':>9} |"
            print(row)
        
        return results, predictions
    
    def predict_single(self, house_data):
        """
        Make predictions for a single house.
        
        Args:
            house_data: Dictionary with house features
            
        Returns:
            Dictionary with predictions from all models
        """
        # Convert to DataFrame
        df = pd.DataFrame([house_data])
        
        try:
            X, _ = self.preprocess_data(df)
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            return {}
        
        predictions = {}
        for model_name, model in self.models.items():
            try:
                y_pred_log = model.predict(X)[0]
                y_pred = np.expm1(y_pred_log)
                predictions[model_name] = {
                    'price': y_pred,
                    'log_price': y_pred_log
                }
            except Exception as e:
                print(f"❌ Error predicting with {model_name}: {e}")
        
        return predictions
    
    def compare_models(self, test_results):
        """
        Compare model performance and recommend best model.
        
        Args:
            test_results: Results from test_models() function
        """
        if not test_results:
            print("❌ No test results to compare")
            return
        
        print(f"\n🏆 MODEL COMPARISON AND RANKING")
        print("=" * 60)
        
        # Rank by R² score (higher is better)
        r2_ranking = sorted(test_results.items(), key=lambda x: x[1]['r2_score'], reverse=True)
        print("📊 Ranking by R² Score:")
        for i, (model_name, metrics) in enumerate(r2_ranking, 1):
            print(f"   {i}. {model_name:20} R² = {metrics['r2_score']:.3f}")
        
        # Rank by MAE (lower is better)
        mae_ranking = sorted(test_results.items(), key=lambda x: x[1]['mae'])
        print(f"\n📊 Ranking by MAE (Lower is Better):")
        for i, (model_name, metrics) in enumerate(mae_ranking, 1):
            print(f"   {i}. {model_name:20} MAE = {metrics['mae']:.2f}")
        
        # Best model overall (you can adjust this logic)
        best_model = r2_ranking[0][0]
        print(f"\n🥇 RECOMMENDED MODEL: {best_model}")
        print(f"   This model has the highest R² score of {r2_ranking[0][1]['r2_score']:.3f}")


def load_latest_model(model_dir="models"):
    """Load the most recent model metadata."""
    latest_metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    
    if os.path.exists(latest_metadata_path):
        return latest_metadata_path
    else:
        # Find the most recent metadata file
        metadata_files = [f for f in os.listdir(model_dir) if f.startswith("model_metadata_") and f.endswith(".json")]
        if metadata_files:
            latest_file = sorted(metadata_files)[-1]
            return os.path.join(model_dir, latest_file)
        else:
            raise FileNotFoundError(f"No model metadata found in {model_dir}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test saved house price prediction models")
    parser.add_argument("--model_dir", default="models", help="Directory containing saved models")
    parser.add_argument("--data_path", help="Path to test data JSON file")
    parser.add_argument("--single_prediction", help="JSON string for single house prediction")
    
    args = parser.parse_args()
    
    try:
        # Load the latest model
        metadata_path = load_latest_model(args.model_dir)
        print(f"🔍 Using model metadata: {metadata_path}")
        
        # Initialize model tester
        tester = ModelTester(metadata_path)
        
        # Print model information
        tester.print_model_info()
        
        # Test on data if provided
        if args.data_path:
            if os.path.exists(args.data_path):
                results, predictions = tester.test_models(args.data_path)
                tester.compare_models(results)
            else:
                print(f"❌ Test data file not found: {args.data_path}")
        
        # Single prediction if provided
        if args.single_prediction:
            try:
                house_data = json.loads(args.single_prediction)
                print(f"\n🏠 Single House Prediction:")
                print(f"Input: {house_data}")
                
                predictions = tester.predict_single(house_data)
                print(f"\nPredictions:")
                for model_name, pred in predictions.items():
                    print(f"   {model_name}: {pred['price']:,.2f} TND")
                    
            except json.JSONDecodeError:
                print("❌ Invalid JSON for single prediction")
        
        # If no specific test requested, show usage
        if not args.data_path and not args.single_prediction:
            print(f"\n💡 Usage Examples:")
            print(f"   Test on data: python test_models.py --data_path data/test_data.json")
            print(f"   Single prediction: python test_models.py --single_prediction '{{\"room_count\": 3, \"land_area\": 150}}'")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
