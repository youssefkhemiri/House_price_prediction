import pandas as pd
import os
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import regex as re
import numpy as np

def load_dataframe(json_path):
    """Load DataFrame from JSON file, handling both array and JSON lines formats."""
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return None
    try:
        df = pd.read_json(json_path)
    except ValueError:
        df = pd.read_json(json_path, lines=True)
    return df

def check_price_column(df):
    """Check for missing, non-numeric, or problematic values in the 'price' column."""
    if 'price' not in df.columns:
        print("'price' column not found in the data.")
        return
    # Check for missing values
    missing = df['price'].isnull().sum()
    print(f"Missing values in 'price': {missing}")
    if missing > 0:
        print("Sample rows with missing 'price':")
        print(df.loc[df['price'].isnull(), ['price']].head())
    # Check for non-numeric types
    non_numeric = df[~df['price'].apply(lambda x: isinstance(x, (int, float)) or pd.isnull(x))]
    if not non_numeric.empty:
        print(f"Rows with non-numeric 'price' values: {len(non_numeric)}")
        print(non_numeric[['price']])
    else:
        print("All non-missing 'price' values are numeric.")


load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = KEY)
#client = OpenAI()

def call_openai_model(prompt, modell="gpt-4o"):

    messages = [
        #{"role": "assistant", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
        ]

    response = client.chat.completions.create(
        model = modell,
        messages = messages
        )
    return response.choices[0].message.content



def display_sample_descriptions(df, n=5):
    """Display a sample of the 'description' column if it exists."""
    if 'description' in df.columns:
        print("Sample descriptions:")
        print(df['description'].dropna().head(n))
    else:
        print("No 'description' column found.")


def extract_price_from_description(df, model="gpt-4o", n=5):
    """Attempt to extract price from the description column using OpenAI, only for rows where 'price' is null/NaN."""
    if 'description' not in df.columns or 'price' not in df.columns:
        print("'description' or 'price' column not found.")
        return 0
    # Filter for rows where price is null/NaN
    missing_price_df = df[df['price'].isnull() & df['description'].notnull()]
    sample = missing_price_df['description'].head(n)
    success_count = 0
    for desc in sample:
        # Ensure desc is a string
        if isinstance(desc, list):
            desc_str = ' '.join(str(x) for x in desc)
        else:
            desc_str = str(desc)
        prompt = (
            "Extract the price from this real estate listing description. "
            "Only return the price as a number in thousands (e.g., 250 for 250,000). "
            "If the price is not present, return nothing. Description: " + desc_str
        )
        try:
            result = call_openai_model(prompt, modell=model)
            # Try to parse the result as a float or int
            price = None
            try:
                price = float(result.replace(',', '').strip())
            except ValueError:
                pass
            if price is not None:
                print(f"Success: Extracted price {price} (thousands) from description: {desc_str}")
                success_count += 1
            else:
                print(f"Failed: Could not extract price. Model output: {result}")
        except Exception as e:
            print(f"OpenAI call failed: {e}")
    print(f"Number of successful price extractions: {success_count} out of {len(sample)}")
    return success_count

def delete_nan_prices(df):
    """Remove all rows where the 'price' column is NaN and return the cleaned DataFrame."""
    if 'price' not in df.columns:
        print("'price' column not found.")
        return df
    cleaned_df = df.dropna(subset=['price'])
    print(f"Removed {len(df) - len(cleaned_df)} rows with NaN prices.")
    return cleaned_df

def print_unique_property_types(df):
    """Print the unique values in the 'property_type' column."""
    if 'property_type' in df.columns:
        unique_types = df['property_type'].dropna().unique()
        print(f"Unique property types ({len(unique_types)}): {unique_types}")
    else:
        print("'property_type' column not found.")

def remove_unwanted_property_types(df):
    """Remove rows with unwanted property_type values and print how many were deleted."""
    unwanted_types = [
        'Terrain', 'Local commercial', 'Bureau', 'Surfaces', 'Autre', 'Terrain nu',
        'Place de parc', None, 'Terrain agricole', 'Commerces', 'Terrains', 'Lots',
        'Bureaux', 'Locaux commerciaux', 'Terrain agr', 'Fond de Com', 'Gérance lib', 'Atelier'
    ]
    before = len(df)
    cleaned_df = df[~df['property_type'].isin(unwanted_types)]
    after = len(cleaned_df)
    print(f"Removed {before - after} rows with unwanted property_type values.")
    return cleaned_df

def print_unique_climate_values(df):
    """Print unique values in 'heating' and 'air_conditioning' columns."""
    for col in ['heating', 'air_conditioning']:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"Unique values in '{col}' ({len(unique_vals)}): {unique_vals}")
        else:
            print(f"'{col}' column not found.")

def print_unique_location_values(df):
    """Print unique values in 'address', 'governorate', 'delegation', and 'locality' columns."""
    for col in ['address', 'governorate', 'delegation', 'locality']:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"Unique values in '{col}' ({len(unique_vals)}): {unique_vals[:10]} ...")
        else:
            print(f"'{col}' column not found.")

def extract_json_from_text(text):
    """Extract the first JSON object from a string, even if surrounded by other text or markdown code blocks."""
    # Try to find the first {...} block
    match = re.search(r'({(?:[^{}]|(?R))*})', text, re.DOTALL)
    if match:
        return match.group(1)
    # Optionally, try to match a JSON array if object not found
    match = re.search(r'(\[(?:[^\[\]]|(?R))*\])', text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def convert_json_safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    return val

def extract_features_with_openai(df, model="gpt-4o", n=5000, output_json_path="data/processed/enriched_real_estate_data.json"):
    """
    For the first n rows, use OpenAI to extract structured features from description and address.
    Update only if the target columns are NULL/empty, keep price and description, and save the result as a new JSON file.
    Also save the original data for comparison.
    Each processed row is appended to the output file as NDJSON.
    """
    prompt_template = (
        "You are an expert real estate assistant. Based on the provided unstructured listing description and partial address, extract the following structured information:\n"
        "1. `heating`: 1 if the house has heating mentioned (any type), 0 otherwise.  \n"
        "2. `air_conditioning`: 1 if air conditioning (e.g., AC, climatization) is mentioned, 0 otherwise.  \n"
        "3. Address fields (extract separately if possible):\n"
        "   - `address`: the most complete address possible  \n"
        "   - `governorate`  \n"
        "   - `delegation`  \n"
        "   - `locality`  \n"
        "4. Features (binary: 1 if present, 0 if not mentioned):\n"
        "   - `has_garage`  \n"
        "   - `has_garden`  \n"
        "   - `has_pool`  \n"
        "   - `has_balcony`  \n"
        "   - `has_terrace`  \n"
        "5. Room details:\n"
        "   - `room_count`: number of bedrooms (if a range or unclear, give your best guess)  \n"
        "   - `bathroom_count`: number of bathrooms  \n"
        "6. `land_area`: surface area of the land in square meters (if mentioned if not default to 0)  \n"
        "7. Finally, rate the property from 0 to 10 in a field called `quality_score`, where:\n"
        "   - 0 means the house is in poor condition or missing key features  \n"
        "   - 10 means the house is in excellent condition, well-equipped, and high price is justified  \n"
        "Respond only in clean JSON format like this:\n{...}"
    )
    # Columns to update
    target_cols = [
        "heating", "air_conditioning", "address", "governorate", "delegation", "locality",
        "has_garage", "has_garden", "has_pool", "has_balcony", "has_terrace", "room_count", "bathroom_count", "quality_score", "land_area"
    ]
    df = df.copy()
    # Ensure 'quality_score' column exists
    if 'quality_score' not in df.columns:
        df['quality_score'] = 0
    # Save the original data for comparison (include description)
    output_cols = target_cols + ["price", "description"]
    output_cols = [col for col in output_cols if col in df.columns]
    df.head(n)[output_cols].to_json("before_enriched_real_estate_data.json", orient="records", force_ascii=False)
    print(f"Saved original data for {n} rows to before_enriched_real_estate_data.json")
    processed = 0
    with open(output_json_path, "a", encoding="utf-8") as f:
        for idx, row in df.head(n).iterrows():
            print(f"Processing row {idx} ...")
            desc = row.get("description", "")
            addr = row.get("address", "")
            if not desc and not addr:
                continue
            prompt = f"{prompt_template}\nDescription: {desc}\nAddress: {addr}"
            try:
                response = call_openai_model(prompt, modell=model)
                # Try to extract and parse JSON from response
                features = None
                try:
                    json_str = extract_json_from_text(response)
                    if json_str:
                        features = json.loads(json_str)
                        print(f"Extracted quality_score: {features.get('quality_score', 'NOT FOUND')}")
                    else:
                        raise ValueError("No JSON found in response")
                except Exception as e:
                    print(f"Row {idx}: Failed to parse JSON from OpenAI response. Skipping. Response: {response}")
                    continue
                # Update only if column is null/empty
                for col in target_cols:
                    if col == 'quality_score':
                        df.at[idx, col] = features.get('quality_score', 0)
                    elif (col in df.columns and (pd.isnull(row.get(col)) or row.get(col) == "")) or (col not in df.columns):
                        df.at[idx, col] = features.get(col, row.get(col))
                # Ensure 'quality_score' is present in the output
                if 'quality_score' not in df.columns or pd.isnull(df.at[idx, 'quality_score']):
                    df.at[idx, 'quality_score'] = features.get('quality_score', 0)
                processed_row = {col: convert_json_safe(df.at[idx, col]) for col in output_cols}
                f.write(json.dumps(processed_row, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as e:
                print(f"Row {idx}: OpenAI call failed: {e}")
    print(f"Saved enriched data for {processed} rows to {output_json_path}")
    return df

def remove_duplicate_descriptions(df):
    """Remove rows with duplicate 'description' values, keeping the first occurrence."""
    before = len(df)
    cleaned_df = df.drop_duplicates(subset=['description'], keep='first')
    after = len(cleaned_df)
    print(f"Removed {before - after} rows with duplicate descriptions.")
    return cleaned_df


# To use this function, uncomment below in main:
# 


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Enrich real estate data with OpenAI")
    parser.add_argument(
        "--input",
        default=r"data\raw\combined_data.json",
        help="Input JSON file path"
    )
    parser.add_argument(
        "--output", 
        default=r"data\processed\enriched_real_estate_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of rows to process"
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip data cleaning steps"
    )
    
    args = parser.parse_args()
    
    print("🏠 House Price Prediction - Data Enrichment")
    print("=" * 50)
    print(f"📥 Input file: {args.input}")
    print(f"📤 Output file: {args.output}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Processing limit: {args.limit} rows")
    
    # Load data
    df = load_dataframe(args.input)
    if df is None:
        print("❌ Failed to load data")
        return 1
        
    print(f"📊 Loaded {len(df)} records")
    print("Columns:", df.columns.tolist())
    print("--------------------------------")
    
    if not args.skip_cleanup:
        # Data cleaning pipeline
        print("🧹 Cleaning data...")
        
        # Remove NaN prices
        df = delete_nan_prices(df)
        check_price_column(df)
        print("--------------------------------")
        
        # Remove unwanted property types
        df = remove_unwanted_property_types(df)
        print_unique_property_types(df)
        print("--------------------------------")
        
        # Show location info
        print_unique_location_values(df)
        print("--------------------------------")
        
        # Remove duplicates
        df = remove_duplicate_descriptions(df)
        print("--------------------------------")
    
    # Run OpenAI enrichment
    print(f"🤖 Starting OpenAI enrichment with {args.model}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Clear output file if it exists
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f"🗑️  Cleared existing output file: {args.output}")
    
    try:
        enriched_df = extract_features_with_openai(
            df, 
            model=args.model, 
            n=args.limit,
            output_json_path=args.output
        )
        
        print(f"\n🎉 Data enrichment completed successfully!")
        print(f"📄 Enriched data saved to: {args.output}")
        
        # Display summary
        if os.path.exists(args.output):
            with open(args.output, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"📊 Total enriched records: {lines}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Enrichment failed: {e}")
        return 1


        

if __name__ == "__main__":
    exit(main())

