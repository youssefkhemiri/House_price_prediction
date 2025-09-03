#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Combiner Script for House Price Prediction

This script combines multiple JSON data sources from the raw data directory
into a unified dataset for processing and model training.

Features:
- Combines data from multiple scrapers (Menzili, Mubawab, etc.)
- Handles different JSON formats (array and NDJSON)
- Removes duplicates based on description similarity
- Adds source tracking for data lineage
- Validates and cleans combined data
"""

import pandas as pd
import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import hashlib


def load_json_file(file_path: str) -> List[Dict[Any, Any]]:
    """
    Load JSON data from file, handling both array and NDJSON formats.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return []
    
    try:
        # First try to load as regular JSON array
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]  # Single object
                
    except json.JSONDecodeError:
        # If that fails, try NDJSON format
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
            return []


def get_description_hash(description: str) -> str:
    """Generate a hash for description to help identify duplicates."""
    if not description:
        return ""
    # Normalize text: lowercase, remove extra spaces
    normalized = ' '.join(str(description).lower().split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def combine_raw_data(raw_data_dir: str, output_path: str, exclude_files: List[str] = None) -> int:
    """
    Combine all JSON files from raw data directory into a unified dataset.
    
    Args:
        raw_data_dir: Directory containing raw JSON files
        output_path: Path for combined output file
        exclude_files: List of filenames to exclude
        
    Returns:
        Number of records in combined dataset
    """
    if exclude_files is None:
        exclude_files = ['.gitkeep', 'combined_data.json']
    
    raw_dir = Path(raw_data_dir)
    if not raw_dir.exists():
        print(f"❌ Raw data directory not found: {raw_data_dir}")
        return 0
    
    print(f"🔍 Scanning directory: {raw_data_dir}")
    
    combined_data = []
    source_stats = {}
    description_hashes = set()
    
    # Find all JSON files
    json_files = [f for f in raw_dir.glob("*.json") if f.name not in exclude_files]
    
    if not json_files:
        print("⚠️  No JSON files found in raw data directory")
        return 0
    
    for json_file in json_files:
        print(f"📥 Loading: {json_file.name}")
        data = load_json_file(str(json_file))
        
        if not data:
            continue
            
        # Add source information and process records
        source_name = json_file.stem  # filename without extension
        processed_records = 0
        duplicate_records = 0
        
        for record in data:
            # Add source tracking
            record['data_source'] = source_name
            record['source_file'] = json_file.name
            
            # Check for duplicates based on description
            description = record.get('description', '')
            desc_hash = get_description_hash(description)
            
            if desc_hash and desc_hash in description_hashes:
                duplicate_records += 1
                continue  # Skip duplicate
            
            if desc_hash:
                description_hashes.add(desc_hash)
            
            combined_data.append(record)
            processed_records += 1
        
        source_stats[source_name] = {
            'total_loaded': len(data),
            'processed': processed_records,
            'duplicates_skipped': duplicate_records
        }
        
        print(f"   ✅ {processed_records} records added, {duplicate_records} duplicates skipped")
    
    # Save combined data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📊 COMBINATION SUMMARY")
    print("=" * 50)
    for source, stats in source_stats.items():
        print(f"{source:20}: {stats['processed']:5d} records ({stats['duplicates_skipped']} duplicates skipped)")
    
    print("-" * 50)
    print(f"{'TOTAL':20}: {len(combined_data):5d} records")
    print(f"💾 Combined data saved to: {output_path}")
    
    return len(combined_data)


def validate_combined_data(data_path: str) -> bool:
    """
    Validate the combined dataset for common issues.
    
    Args:
        data_path: Path to combined data file
        
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n🔍 Validating combined data: {data_path}")
    
    try:
        df = pd.read_json(data_path)
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['price', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        # Check data quality
        print("\n📈 Data Quality Summary:")
        print(f"   Price column: {df['price'].notna().sum():,} non-null values")
        print(f"   Description column: {df['description'].notna().sum():,} non-null values")
        
        if 'data_source' in df.columns:
            print(f"   Data sources: {df['data_source'].value_counts().to_dict()}")
        
        # Check for basic issues
        price_issues = df['price'].isna().sum()
        if price_issues > 0:
            print(f"⚠️  {price_issues} records with missing prices")
        
        desc_issues = df['description'].isna().sum()
        if desc_issues > 0:
            print(f"⚠️  {desc_issues} records with missing descriptions")
        
        print("✅ Validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Combine raw JSON data files")
    parser.add_argument(
        "--raw_dir", 
        default="data/raw",
        help="Directory containing raw JSON files"
    )
    parser.add_argument(
        "--output",
        default="data/raw/combined_data.json", 
        help="Output path for combined data"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the combined data after creation"
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[".gitkeep", "combined_data.json", "backups"],
        help="Files/directories to exclude"
    )
    
    args = parser.parse_args()
    
    print("🏠 House Price Prediction - Data Combiner")
    print("=" * 50)
    
    # Combine the data
    record_count = combine_raw_data(
        raw_data_dir=args.raw_dir,
        output_path=args.output,
        exclude_files=args.exclude
    )
    
    if record_count == 0:
        print("❌ No data was combined")
        return 1
    
    # Validate if requested
    if args.validate:
        if not validate_combined_data(args.output):
            print("❌ Validation failed")
            return 1
    
    print(f"\n🎉 Successfully combined {record_count:,} records!")
    return 0


if __name__ == "__main__":
    exit(main())
