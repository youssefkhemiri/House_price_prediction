# menzili_scraper.py
from __future__ import annotations
import re, json, math
from datetime import datetime, timezone
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

PHONE_RE = re.compile(r"(?:\+?216[-.\s]?)?(\d{2}[-.\s]?\d{3}[-.\s]?\d{3,4}|\d{8})")

def _clean_text(x: str | None) -> str | None:
    if not x: return None
    t = re.sub(r"\s+", " ", x).strip()
    return t or None

def _to_int(s: str | None) -> int | None:
    if not s: return None
    s = s.replace("\xa0", " ").replace(" ", "").replace(" ", "")
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s.isdigit() else None

def _price_and_currency(raw: str | None):
    if not raw: return None, None
    t = raw.replace("\xa0", " ").strip()
    # examples: "990 000 DT", "320,000 TND", "1 200 000 Dinars"
    cur = None
    if re.search(r"\bDT\b|\bTND\b|Dinar", t, re.I): cur = "TND"
    nums = re.findall(r"\d[\d\s.,]*", t)
    if nums:
        n = nums[0].replace(" ", "").replace("\u202f", "").replace(",", "")
        try:
            return int(float(n)), cur
        except:
            pass
    return None, cur

def _epoch_ms_from_date_attr(iso_date: str | None) -> int | None:
    if not iso_date: return None
    try:
        dt = datetime.fromisoformat(iso_date).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

def _extract_address_parts(raw_addr: str | None):
    """
    Expects something like: 'www.visavis-immobilier.com, Djerba - Midoun, Médenine'
    Returns (address, governorate, delegation, locality)
    """
    if not raw_addr: 
        return None, "nan", None, None
    # Drop leading URL if present
    parts = [p.strip() for p in raw_addr.split(",") if _clean_text(p)]
    # Heuristic: last = governorate, previous may contain 'delegation - locality'
    governorate = parts[-1] if parts else None
    delegation, locality = None, None
    if len(parts) >= 2:
        mid = parts[-2]
        if " - " in mid:
            a, b = [s.strip() for s in mid.split(" - ", 1)]
            delegation, locality = a or None, b or None
        else:
            delegation = mid
    # Build a concise address string (drop the leading domain-ish chunk if present)
    addr_pieces = []
    for p in parts:
        if re.match(r"^https?://|www\.", p, re.I): 
            continue
        addr_pieces.append(p)
    address = ", ".join(addr_pieces) or None
    return address, governorate or "nan", delegation, locality

def _bool_from_options(options: set[str], key_variants: list[str]) -> bool | None:
    for k in key_variants:
        for opt in options:
            if k.lower() in opt.lower():
                return True
    return None  # keep None if not mentioned at all

def _detect_transaction_type(title: str | None, descr_list: list[str]):
    hay = " ".join([t for t in [title] + descr_list if t]) if (title or descr_list) else ""
    if re.search(r"\b(?:a|à)\s*vendre\b", hay, re.I): return "sale"
    if re.search(r"\b(?:a|à)\s*louer\b", hay, re.I): return "rent"
    return None

def scrape_menzili(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # Title & top block
    h1 = soup.select_one(".product-title-h1 h1[itemprop='name']")
    title = _clean_text(h1.get_text()) if h1 else None

    # Address line under title
    addr_p = soup.select_one(".product-title-h1 p")
    raw_addr = _clean_text(addr_p.get_text()) if addr_p else None
    address, governorate, delegation, locality = _extract_address_parts(raw_addr)

    # Price
    price_p = soup.select_one(".product-price p")
    price_raw = _clean_text(price_p.get_text(" ")) if price_p else None
    price, currency = _price_and_currency(price_raw)

    # Ref & date
    ref_block = soup.select_one(".block-ref")
    listing_id = None
    listing_date_ms = None
    if ref_block:
        ref_text = ref_block.get_text(" ")
        m = re.search(r"Rèf:\s*([0-9]+)", ref_text, re.I)
        if m: listing_id = m.group(1)
        time_tag = ref_block.select_one("time[itemprop='datePublished']")
        if time_tag and time_tag.has_attr("datetime"):
            listing_date_ms = _epoch_ms_from_date_attr(time_tag["datetime"])

    # Photos
    photos = []
    for im in soup.select(".slider-product img[src]"):
        src = im.get("src")
        if src and "upload/photos" in src:
            photos.append(src)
    photos = list(dict.fromkeys(photos)) or None

    # Description (split by <br>)
    desc_p = soup.select_one(".block-descr p[itemprop='text']")
    description = []
    if desc_p:
        # Convert <br> to newlines then split
        html = str(desc_p).replace("<br/>", "\n").replace("<br>", "\n").replace("<br />", "\n")
        text = BeautifulSoup(html, "lxml").get_text("\n")
        description = [s.strip() for s in text.split("\n") if _clean_text(s)]
    if not description:
        description = []

    # Details (rooms, bathrooms, pieces, surfaces)
    details = { _clean_text(span.get_text(":")): _clean_text(span.find_next("strong").get_text() if span.find_next("strong") else None)
                for span in soup.select(".block-detail .block-over span") if span.find_next("strong") }
    # Normalize keys
    room_count = _to_int(details.get("Chambres :"))
    bathroom_count = _to_int(details.get("Salle de bain :"))
    piece_total = _to_int(details.get("Piéces (Totale) :"))
    living_area = None
    land_area = None
    if details.get("Surf habitable :"):
        living_area = _to_int(details["Surf habitable :"])
    if details.get("Surf terrain :"):
        land_area = _to_int(details["Surf terrain :"])

    # Options → boolean features
    options_texts = set(_clean_text(x.get_text()) for x in soup.select(".span-opts strong") if _clean_text(x.get_text()))
    has_garage = _bool_from_options(options_texts, ["Garage"])
    has_garden = _bool_from_options(options_texts, ["Jardin"])
    has_pool = _bool_from_options(options_texts, ["Piscine"])
    has_balcony = _bool_from_options(options_texts, ["Balcon"])
    has_terrace = _bool_from_options(options_texts, ["Terrasse", "Terrasses"])
    heating = _bool_from_options(options_texts, ["Chauffage", "Chauffage électriques"])
    air_conditioning = _bool_from_options(options_texts, ["Climatisation", "Climatise"])
    furnished = _bool_from_options(options_texts, ["Meublé", "Meuble"])

    # Phones (from description & page)
    page_text = soup.get_text(" ")
    phones = list(sorted(set(m.group(0).strip() for m in PHONE_RE.finditer(page_text))))

    # Agency / contact name (best-effort; often elsewhere on page)
    agency = None
    for lbl in ["Agence", "Agence immobili", "Visavis Immobilier", "Immo"]:
        if re.search(lbl, page_text, re.I):
            agency = "Visavis Immobilier" if "Visavis" in lbl or re.search("Visavis", page_text, re.I) else None
            if agency: break

    # Property type (best-effort heuristics)
    property_type = None
    for t, pats in {
        "Maison": [r"\bmaison\b", r"\bvilla\b", r"\bpavillon\b"],
        "Appartement": [r"\bappartement\b"],
        "Terrain": [r"\bterrain\b"],
        "Duplex": [r"\bduplex\b"],
        "Studio": [r"\bstudio\b"],
    }.items():
        if any(re.search(p, (title or "") + " " + " ".join(description), re.I) for p in pats):
            property_type = t
            break

    transaction_type = _detect_transaction_type(title, description)

    out = {
        "url": url,
        "id": listing_id,
        "source": "menzili",
        "property_type": property_type,
        "price": str(price) if price is not None else "nan",
        "currency": currency,
        "description": description,
        "address": address,
        "governorate": governorate if governorate else "nan",
        "delegation": delegation,
        "locality": locality,
        "postal_code": None,              # not present on page
        "living_area": str(living_area) if living_area is not None else "nan",
        "land_area": land_area,
        "room_count": room_count,
        "bathroom_count": bathroom_count,
        "construction_year": None,
        "floor": None,
        "has_garage": has_garage,
        "has_garden": has_garden,
        "has_pool": has_pool,
        "has_balcony": has_balcony,
        "has_terrace": has_terrace,
        "heating": heating,
        "air_conditioning": air_conditioning,
        "furnished": furnished,
        "phone": phones,
        "agency": agency,
        "contact_name": None,
        "listing_date": listing_date_ms,
        "last_updated": None,
        "photos": photos,
        "features": list(options_texts) if options_texts else None,
        "condition": None,
        "transaction_type": transaction_type,
    }
    return out

if __name__ == "__main__":
    test_url = "https://www.menzili.tn/annonce/belle-villa-avec-piscine-privEe-en-zone-touristique-djerba-medenine-djerbamidoun-142024"
    data = scrape_menzili(test_url)
    print(json.dumps(data, ensure_ascii=False, indent=2))
