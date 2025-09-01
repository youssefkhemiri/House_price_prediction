# any_website_scraper.py
# pip install openai>=1.40.0 requests beautifulsoup4 lxml regex
import os, json, math, time, urllib.parse, re as stdre
import regex as rx  # supports recursive regex for balanced braces
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Optional, Dict, Any, List

OPENAI_MODEL_PRIMARY = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # supports structured outputs
OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o")  # JSON mode fallback
TIMEOUT = (10, 25)  # (connect, read)

FIELDS = [
    "url","id","source","property_type","price","currency","description","address",
    "governorate","delegation","locality","postal_code","living_area","land_area",
    "room_count","bathroom_count","construction_year","floor","has_garage","has_garden",
    "has_pool","has_balcony","has_terrace","heating","air_conditioning","furnished",
    "phone","agency","contact_name","listing_date","last_updated","photos","features",
    "condition","transaction_type"
]

# -------------------
# Fetch & HTML reduce
# -------------------
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36"),
        "Accept-Language": "en;q=0.9,fr;q=0.8"
    }
    r = requests.get(url, headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def reduce_html(html: str, keep_chars: int = 120_000) -> str:
    """
    Strip scripts/styles, keep visible text + key attributes to stay within token limits.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Pull meta worth keeping
    metas = []
    for m in soup.find_all("meta"):
        for k in ("name", "property"):
            if m.get(k, "").lower() in {"description","og:description","og:title","twitter:description","twitter:title"}:
                val = m.get("content")
                if val:
                    metas.append(f"{m.get(k)}: {val}")

    # Pull img URLs (useful for photos)
    imgs = [img.get("src") for img in soup.find_all("img") if img.get("src")]
    imgs = list(dict.fromkeys(imgs))[:100]  # unique, limit

    text = soup.get_text("\n")
    bundle = "\n\n---META---\n" + "\n".join(metas) + "\n\n---IMAGES---\n" + "\n".join(imgs) + "\n\n---TEXT---\n" + text
    return bundle[:keep_chars]

# ---------------------------------------
# Model prompt (with sample & strict spec)
# ---------------------------------------
SAMPLE_JSON = {
  "url": "https://example.com/listing/123",
  "id": "123",
  "source": "example",
  "property_type": "Apartment",
  "price": 1070000,
  "currency": "TND",
  "description": ["Beautiful apartment...", "Great location..."],
  "address": "Sidi Daoued, La Marsa",
  "governorate": "Tunis",
  "delegation": "La Marsa",
  "locality": "Sidi Daoued",
  "postal_code": "2078",
  "living_area": 287,
  "land_area": 40,
  "room_count": 3,
  "bathroom_count": 3,
  "construction_year": 2015,
  "floor": 2,
  "has_garage": True,
  "has_garden": True,
  "has_pool": False,
  "has_balcony": True,
  "has_terrace": True,
  "heating": "Central heating",
  "air_conditioning": "Yes",
  "furnished": False,
  "phone": ["+21612345678"],
  "agency": "Immo Expert",
  "contact_name": "John Doe",
  "listing_date": 1730764800000,
  "last_updated": 1731628800000,
  "photos": ["https://example.com/p1.jpg","https://example.com/p2.jpg"],
  "features": ["Elevator","Concierge","Double glazing","Reinforced door"],
  "condition": "Good condition",
  "transaction_type": "sale"
}

def build_user_prompt(url: str, html_snippet: str) -> str:
    fields_list = ", ".join(FIELDS)
    return f"""
You are a strict data extraction engine.

Task: Read the provided real estate listing HTML and return a SINGLE JSON object with the following keys:
[{fields_list}]

Rules:
- If a field is missing or cannot be inferred, set it to null.
- Numbers must be numeric (not strings). Counts/areas as integers if whole, otherwise numbers.
- Booleans must be true/false (not "yes"/"no").
- Arrays required for: description, phone, photos, features. If none found, use [] (empty array).
- "source" should be the website/app source (usually the domain or brand), if inferable, else null.
- "currency" should be ISO-like symbol or 3-4 letter code if possible (e.g., "TND", "EUR"), else null.
- "transaction_type" one of: "sale", "rent", "lease", "seasonal", "vacation", "unknown" (or null if not clear).
- "listing_date" and "last_updated" should be Unix epoch **milliseconds** if present/inferable; else null.
- Normalize "property_type" (Apartment, House, Villa, Terrain/Land, Studio, Office, Commercial, Farm, Duplex, Penthouse, etc).

Return ONLY the JSON object. No markdown, no comments, no prose.

Example shape (mock data):
{json.dumps(SAMPLE_JSON, ensure_ascii=False, indent=2)}

URL: {url}

<BEGIN_HTML>
{html_snippet}
<END_HTML>
""".strip()

# ------------------------------
# JSON Schema for Structured Out
# ------------------------------
def extraction_json_schema() -> Dict[str, Any]:
    def t_str(): return {"type": ["string", "null"]}
    def t_num(): return {"type": ["number", "null"]}
    def t_int(): return {"type": ["integer", "null"]}
    def t_bool(): return {"type": ["boolean", "null"]}
    def t_str_arr(): return {"type": "array", "items": {"type": "string"}}
    def t_str_arr_or_null(): return {"anyOf": [t_str_arr(), {"type": "null"}]}

    return {
        "name": "listing_schema",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": t_str(), "id": t_str(), "source": t_str(), "property_type": t_str(),
                "price": t_num(), "currency": t_str(), "description": t_str_arr(),
                "address": t_str(), "governorate": t_str(), "delegation": t_str(),
                "locality": t_str(), "postal_code": t_str(), "living_area": t_num(),
                "land_area": t_num(), "room_count": t_int(), "bathroom_count": t_int(),
                "construction_year": t_int(), "floor": t_int(), "has_garage": t_bool(),
                "has_garden": t_bool(), "has_pool": t_bool(), "has_balcony": t_bool(),
                "has_terrace": t_bool(), "heating": t_str(), "air_conditioning": t_str(),
                "furnished": t_bool(), "phone": t_str_arr(), "agency": t_str(),
                "contact_name": t_str(), "listing_date": t_int(), "last_updated": t_int(),
                "photos": t_str_arr_or_null(), "features": t_str_arr_or_null(),
                "condition": t_str(), "transaction_type": t_str()
            },
            "required": FIELDS
        },
        "strict": True
    }

# --------------------------------------------------------
# Regex-based JSON extractor (balanced braces via `regex`)
# --------------------------------------------------------
def extract_json_with_regex(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object from `text` using a recursive regex
    that matches balanced braces. Requires the `regex` module (pip install regex).
    Returns the JSON string, or None if not found.
    """
    # (?s) = DOTALL; (?R) = recursive subpattern (only in `regex`, not stdlib `re`)
    pattern = rx.compile(r"(?s)\{(?:[^{}]|(?R))*\}")
    m = pattern.search(text)
    return m.group(0) if m else None

def extract_json_fallback(text: str) -> Optional[str]:
    """
    Pure-Python brace counter fallback if `regex` isn't available.
    Grabs the first top-level {...} block.
    """
    start = text.find("{")
    if start == -1: return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None

def parse_model_output_to_dict(raw: str) -> Dict[str, Any]:
    js = extract_json_with_regex(raw) or extract_json_fallback(raw) or raw.strip()
    try:
        data = json.loads(js)
    except Exception:
        # As a last resort, try removing code fences if any
        js2 = stdre.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", js, flags=stdre.DOTALL)
        data = json.loads(js2)
    return data

# ---------------------------------------
# Main extraction using OpenAI + fallback
# ---------------------------------------
def scrape_any_website(url: str) -> Dict[str, Any]:
    html = fetch_html(url)
    snippet = reduce_html(html)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = build_user_prompt(url, snippet)

    # Try Structured Outputs first
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL_PRIMARY,
            input=prompt,
            temperature=0,
            max_output_tokens=1200,
            response_format={
                "type": "json_schema",
                "json_schema": extraction_json_schema()
            },
        )
        # New SDKs expose parsed JSON directly; otherwise use output_text
        # Try to access parsed content if present; else fall back to text.
        raw_text = getattr(resp, "output_text", None)
        if raw_text is None:
            # Try to walk generic structure
            raw_text = json.dumps(resp.output[0].content[0].parsed) if (
                getattr(resp, "output", None) and
                len(resp.output) and
                len(getattr(resp.output[0], "content", [])) and
                hasattr(resp.output[0].content[0], "parsed")
            ) else json.dumps(resp)
        data = parse_model_output_to_dict(raw_text)
    except Exception:
        # Fallback to Chat Completions JSON mode
        chat = client.chat.completions.create(
            model=OPENAI_MODEL_FALLBACK,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise data extractor. Always return a single JSON object and nothing else."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_text = chat.choices[0].message.content
        data = parse_model_output_to_dict(raw_text)

    # Ensure all required keys exist; fill missing with null/empty array as specified
    def ensure_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k in FIELDS:
            if k in d: 
                out[k] = d[k]
            else:
                if k in {"description","phone"}:
                    out[k] = []
                else:
                    out[k] = None
        return out

    return ensure_defaults(data)

# ---------------
# Example usage
# ---------------
if __name__ == "__main__":
    test_url = "https://www.mubawab.tn/en/a/8124624/great-apartment-for-rent-in-sidi-daoued-3-rooms-double-glazed-windows-and-reinforced-door"
    result = scrape_any_website(test_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
















# from __future__ import annotations
# import re, json, math
# from datetime import datetime, timezone
# from urllib.parse import urlparse
# import requests
# from bs4 import BeautifulSoup

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
#                   "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
# }

# def scrape_any_website(url: str) -> dict:
#     """
#     Generic website scraper that attempts to extract property data from any website.
#     Currently returns mock data similar to menzili scraper.
#     TODO: Implement actual generic scraping logic.
#     """
#     # For now, return mock data similar to menzili scraper
#     # In the future, this could use AI/ML to intelligently extract property data
    
#     mock_data = {
#         "url": url,
#         "id": "generic_001",
#         "source": "any_website",
#         "property_type": "Property",
#         "price": "500000",
#         "currency": "TND",
#         "description": [
#             "Generic property extracted from any website",
#             "This is a placeholder implementation",
#             "Future versions will use intelligent extraction"
#         ],
#         "address": "Generic Address",
#         "governorate": "Generic Governorate",
#         "delegation": "Generic Delegation",
#         "locality": "Generic Locality",
#         "postal_code": None,
#         "living_area": "150",
#         "land_area": 200,
#         "room_count": 3,
#         "bathroom_count": 2,
#         "construction_year": None,
#         "floor": None,
#         "has_garage": True,
#         "has_garden": True,
#         "has_pool": False,
#         "has_balcony": True,
#         "has_terrace": False,
#         "heating": True,
#         "air_conditioning": True,
#         "furnished": None,
#         "phone": ["71234567"],
#         "agency": "Generic Agency",
#         "contact_name": None,
#         "listing_date": int(datetime.now().timestamp() * 1000),
#         "last_updated": None,
#         "photos": [
#             "https://via.placeholder.com/400x300/667eea/ffffff?text=Property+Photo+1",
#             "https://via.placeholder.com/400x300/764ba2/ffffff?text=Property+Photo+2"
#         ],
#         "features": [
#             "Generic Feature 1",
#             "Generic Feature 2",
#             "Placeholder Data"
#         ],
#         "condition": "Good",
#         "transaction_type": "sale",
#     }
    
#     return mock_data

# if __name__ == "__main__":
#     test_url = "https://example.com/property"
#     data = scrape_any_website(test_url)
#     print(json.dumps(data, ensure_ascii=False, indent=2))
