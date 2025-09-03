import pandas as pd
import json
import os
import pickle
import joblib
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

def load_ndjson(path):
    # Load NDJSON file into DataFrame
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(data)

def cap_outliers(df, col, lower_quantile=0.01, upper_quantile=0.99):
    lower = df[col].quantile(lower_quantile)
    upper = df[col].quantile(upper_quantile)
    df[col] = np.clip(df[col], lower, upper)
    return df

def save_model_artifacts(models, preprocessors, feature_names, model_metadata, model_dir="models"):
    """
    Save trained models, preprocessors, and metadata to disk.
    
    Args:
        models: dict of trained models {'model_name': model_object}
        preprocessors: dict of preprocessors {'imputer': imputer, 'tfidf': tfidf_vectorizer}
        feature_names: list of feature names
        model_metadata: dict with training metadata
        model_dir: directory to save models
    """
    # Create models directory
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save models
    model_paths = {}
    for name, model in models.items():
        model_path = os.path.join(model_dir, f"{name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        model_paths[name] = model_path
        print(f"✅ Saved {name} model to: {model_path}")
    
    # Save preprocessors
    preprocessor_paths = {}
    for name, preprocessor in preprocessors.items():
        if preprocessor is not None:
            prep_path = os.path.join(model_dir, f"{name}_{timestamp}.joblib")
            joblib.dump(preprocessor, prep_path)
            preprocessor_paths[name] = prep_path
            print(f"✅ Saved {name} preprocessor to: {prep_path}")
    
    # Save feature names and metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'model_paths': model_paths,
        'preprocessor_paths': preprocessor_paths,
        'training_metadata': model_metadata
    }
    
    metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # Also save as latest
    latest_metadata_path = os.path.join(model_dir, "latest_model_metadata.json")
    with open(latest_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_path

def flatten_to_str(x):
    if isinstance(x, list):
        flat = []
        for item in x:
            if isinstance(item, list):
                flat.extend(str(subitem) for subitem in item)
            else:
                flat.append(str(item))
        return ' '.join(flat)
    return str(x)

def clean_data(df, use_tfidf=False):
    df = df.copy()
    df = df[df['price'].notnull() & (df['price'] > 1000)]
    # Prepare description as string for TF-IDF if needed
    if use_tfidf and 'description' in df.columns:
        df['description_str'] = df['description'].apply(flatten_to_str)
    df = df.drop(columns=['address', 'governorate', 'locality', 'delegation', 'description'], errors='ignore')
    # Feature engineering: room_count * land_area
    if 'room_count' in df.columns and 'land_area' in df.columns:
        df['room_area'] = df['room_count'].fillna(0) * df['land_area'].fillna(0)
    # Cap outliers in price and land_area
    df = cap_outliers(df, 'price')
    if 'land_area' in df.columns:
        df = cap_outliers(df, 'land_area')
    # Log-transform price and land_area
    df['log_price'] = np.log1p(df['price'])
    if 'land_area' in df.columns:
        df['log_land_area'] = np.log1p(df['land_area'])
    # Median imputation for missing values
    features = [col for col in df.columns if col not in ['price', 'log_price', 'description_str']]
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])
    # If using TF-IDF, add 'description_str' to the returned features for later processing
    if use_tfidf and 'description_str' in df.columns:
        features.append('description_str')
    return df, features

def train_and_evaluate_models(json_path, use_tfidf=False, save_models=True):
    print(f"Loading data from {json_path} ...")
    df = load_ndjson(json_path)
    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")
    
    # Clean and prepare data
    df, features = clean_data(df, use_tfidf=use_tfidf)
    X = df[features]
    y = df['log_price']
    
    # Initialize preprocessors
    imputer = None
    tfidf_vectorizer = None
    
    # Add TF-IDF features if enabled
    if use_tfidf and 'description_str' in X.columns:
        print("Adding TF-IDF features from description...")
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = tfidf_vectorizer.fit_transform(X['description_str'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(), 
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])], 
            index=df.index
        )
        X = X.drop(columns=['description_str'])
        X = pd.concat([X, tfidf_df], axis=1)
    
    print(f"Training on features: {list(X.columns)}")
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store models and their performance
    models = {}
    results = {}

    # Random Forest with Grid Search
    print("Training Random Forest...")
    rf_params = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    rf = GridSearchCV(
        RandomForestRegressor(random_state=42), 
        rf_params, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    models['random_forest'] = rf.best_estimator_
    results['random_forest'] = {
        'best_params': rf.best_params_,
        'r2_score': r2_score(y_test, y_pred_rf),
        'mae': mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_rf))
    }
    
    print("Random Forest Results:")
    print(f"  Best Params: {rf.best_params_}")
    print(f"  R^2: {results['random_forest']['r2_score']:.3f}")
    print(f"  MAE: {results['random_forest']['mae']:.2f}")

    # XGBoost with Grid Search
    print("Training XGBoost...")
    xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 6]}
    xgbr = GridSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0), 
        xgb_params, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    xgbr.fit(X_train, y_train)
    y_pred_xgb = xgbr.predict(X_test)
    
    models['xgboost'] = xgbr.best_estimator_
    results['xgboost'] = {
        'best_params': xgbr.best_params_,
        'r2_score': r2_score(y_test, y_pred_xgb),
        'mae': mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_xgb))
    }
    
    print("XGBoost Results:")
    print(f"  Best Params: {xgbr.best_params_}")
    print(f"  R^2: {results['xgboost']['r2_score']:.3f}")
    print(f"  MAE: {results['xgboost']['mae']:.2f}")

    # Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    models['linear_regression'] = lr
    results['linear_regression'] = {
        'r2_score': r2_score(y_test, y_pred_lr),
        'mae': mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_lr))
    }
    
    print("Linear Regression Results:")
    print(f"  R^2: {results['linear_regression']['r2_score']:.3f}")
    print(f"  MAE: {results['linear_regression']['mae']:.2f}")

    # Ensemble (VotingRegressor)
    print("Training Ensemble Model...")
    ensemble = VotingRegressor([
        ('rf', rf.best_estimator_), 
        ('xgb', xgbr.best_estimator_), 
        ('lr', lr)
    ])
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    
    models['ensemble'] = ensemble
    results['ensemble'] = {
        'r2_score': r2_score(y_test, y_pred_ens),
        'mae': mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_ens))
    }
    
    print("Ensemble Results:")
    print(f"  R^2: {results['ensemble']['r2_score']:.3f}")
    print(f"  MAE: {results['ensemble']['mae']:.2f}")

    # Preview a few predictions (convert back from log)
    print("\nSample predictions (true vs. RF vs. XGB vs. LR vs. ENS):")
    for i in range(5):
        print(f"True: {np.expm1(y_test.iloc[i]):.2f} | RF: {np.expm1(y_pred_rf[i]):.2f} | XGB: {np.expm1(y_pred_xgb[i]):.2f} | LR: {np.expm1(y_pred_lr[i]):.2f} | ENS: {np.expm1(y_pred_ens[i]):.2f}")

    # Save models if requested
    if save_models:
        print("\n💾 Saving models and artifacts...")
        
        # Prepare preprocessors
        preprocessors = {
            'tfidf_vectorizer': tfidf_vectorizer
        }
        
        # Prepare metadata
        metadata = {
            'data_source': json_path,
            'use_tfidf': use_tfidf,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'results': results,
            'data_shape': X.shape,
            'target_transform': 'log1p'  # We use log1p transformation
        }
        
        # Save everything
        metadata_path = save_model_artifacts(
            models=models,
            preprocessors=preprocessors,
            feature_names=feature_names,
            model_metadata=metadata
        )
        
        print(f"\n🎉 All models saved successfully!")
        print(f"📄 Metadata saved to: {metadata_path}")
        
        return models, preprocessors, metadata_path
    
    return models, {}, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price prediction models")
    parser.add_argument(
        "--data_path",
        default=r"data\processed\enriched_real_estate_data.json",
        help="Path to enriched training data"
    )
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--no_tfidf",
        action="store_true",
        help="Skip TF-IDF training (only train without text features)"
    )
    parser.add_argument(
        "--tfidf_only",
        action="store_true", 
        help="Only train with TF-IDF features"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save models (useful for testing)"
    )
    
    args = parser.parse_args()
    
    print("🏠 House Price Prediction - Model Training")
    print("=" * 50)
    print(f"📊 Data path: {args.data_path}")
    print(f"💾 Model directory: {args.model_dir}")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        print("Please run data enrichment first with: python scripts/enrich_data.py")
        exit(1)
    
    save_models = not args.no_save
    
    if not args.tfidf_only:
        print("\n" + "="*50)
        print("🤖 Training models WITHOUT TF-IDF features")
        print("="*50)
        models, preprocessors, metadata_path = train_and_evaluate_models(
            args.data_path, 
            use_tfidf=False, 
            save_models=save_models
        )
    
    if not args.no_tfidf:
        print("\n" + "="*50)
        print("🤖 Training models WITH TF-IDF features")
        print("="*50)
        models, preprocessors, metadata_path = train_and_evaluate_models(
            args.data_path,
            use_tfidf=True, 
            save_models=save_models
        )
    
    print(f"\n🎉 Training completed successfully!")
    if save_models and metadata_path:
        print(f"📄 Model metadata: {metadata_path}")
        print(f"🔍 Test models with: python scripts/test_models.py")
    
