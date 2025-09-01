# mubawab_scraper.py
from __future__ import annotations
import re, json, html
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

PHONE_RE = re.compile(r"(?:\+?216[-.\s]?)?(\d{2}[-.\s]?\d{3}[-.\s]?\d{3,4}|\d{8})")

def _clean_text(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    t = re.sub(r"\s+", " ", x).strip()
    return t or None

def _to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.replace("\xa0", " ")
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    s = s.replace(" ", "").replace("\u202f", "").replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None

def _price_and_currency(raw: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    if not raw:
        return None, None
    t = raw.replace("\xa0", " ")
    cur = None
    if re.search(r"\b(TND|DT|Dinar|dinars?)\b", t, re.I):
        cur = "TND"
    nums = re.findall(r"\d[\d\s.,]*", t)
    if nums:
        n = nums[0].replace(" ", "").replace("\u202f", "").replace(",", "")
        try:
            return int(float(n)), cur
        except Exception:
            pass
    return None, cur

def _parse_desc_price(description_lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Find 'Price 4,100 TND' style mentions inside description.
    Returns (price, currency) or (None, None)
    """
    blob = " ".join(description_lines)
    m = re.search(r"[Pp]rice\s*[:\-]?\s*([\d\s.,]+)\s*(TND|DT|dinars?)?\b", blob)
    if not m:
        return None, None
    amount = _to_int(m.group(1))
    cur = "TND" if (m.group(2) and re.search(r"TND|DT|dinar", m.group(2), re.I)) else None
    return amount, cur

def _detect_transaction_type(title: Optional[str], descr_list: List[str]) -> Optional[str]:
    hay = " ".join([t for t in [title] + descr_list if t]) if (title or descr_list) else ""
    if re.search(r"\bfor\s*rent\b|\bà\s*louer\b|\ba\s*louer\b", hay, re.I):
        return "rent"
    if re.search(r"\bfor\s*sale\b|\bà\s*vendre\b|\ba\s*vendre\b|\bvendu?\b", hay, re.I):
        return "sale"
    return None

def _photos_from_gallery_json(soup: BeautifulSoup) -> List[str]:
    """
    Mubawab embeds a JSON array of photos in an element like:
    <div id="fl-8124624" pics="[...]">
    """
    urls = []
    for el in soup.select("[id^=fl-][pics]"):
        raw = el.get("pics")
        if not raw:
            continue
        try:
            # html attributes often contain &quot; escaped content
            raw = html.unescape(raw)
            data = json.loads(raw)
            for item in data:
                url = (((item or {}).get("photo") or {}).get("url")) or None
                if url:
                    urls.append(url)
        except Exception:
            continue
    return urls

def _photos_from_imgs(soup: BeautifulSoup) -> List[str]:
    urls = []
    for im in soup.select("#masonryPhoto img[src], #picturesGallery img[src]"):
        src = im.get("src")
        if src and "mubawab-media.com" in src:
            urls.append(src)
    return urls

def _extract_address(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    From <h3 class="greyTit">Sidi Daoued in La Marsa</h3>
    -> locality = 'Sidi Daoued', delegation = 'La Marsa'
    We do not infer governorate; leave as None to avoid over-assuming.
    """
    h = soup.select_one("h3.greyTit")
    if not h:
        return None, None, None, None
    t = _clean_text(h.get_text())
    if not t:
        return None, None, None, None
    m = re.search(r"^(.*?)\s+in\s+(.*)$", t, re.I)
    locality, delegation = None, None
    if m:
        locality = _clean_text(m.group(1))
        delegation = _clean_text(m.group(2))
    address = ", ".join([p for p in [locality, delegation] if p]) or None
    governorate = None
    return address, governorate, delegation, locality

def _bool_from_names(names: List[str], *variants: str) -> Optional[bool]:
    for v in variants:
        for n in names:
            if v.lower() in n.lower():
                return True
    return None

def _collect_feature_labels(soup: BeautifulSoup) -> List[str]:
    labels = []
    # Icon features
    for sp in soup.select(".adFeatures .adFeature span"):
        t = _clean_text(sp.get_text())
        if t:
            labels.append(t)
    # General characteristics - include as features (label: value)
    for box in soup.select(".adMainFeature .adMainFeatureContent"):
        lab = _clean_text((box.select_one(".adMainFeatureContentLabel") or {}).get_text() if box.select_one(".adMainFeatureContentLabel") else None)
        val = _clean_text((box.select_one(".adMainFeatureContentValue") or {}).get_text() if box.select_one(".adMainFeatureContentValue") else None)
        if lab and val:
            labels.append(f"{lab}: {val}")
    return labels

def scrape_mubawab(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    # ID
    adid = None
    id_from_fav = soup.select_one(".favDivPickId[id]")
    if id_from_fav and id_from_fav.has_attr("id") and re.fullmatch(r"\d+", id_from_fav["id"]):
        adid = id_from_fav["id"]
    if not adid:
        hid = soup.select_one("input#adIdLead[value]")
        if hid:
            adid = hid.get("value")

    # Price (big price block)
    price_big = None
    currency_big = None
    price_h3 = soup.select_one(".mainInfoProp h3.orangeTit") or soup.select_one("#fullPicturesHeader .fullPicturesPrice")
    if price_h3:
        price_big, currency_big = _price_and_currency(_clean_text(price_h3.get_text()))

    # Title + description
    title_h1 = soup.select_one(".blockProp h1.searchTitle")
    title = _clean_text(title_h1.get_text()) if title_h1 else None

    desc_p = soup.select_one(".blockProp p")
    description: List[str] = []
    if desc_p:
        # keep line breaks
        html_str = str(desc_p).replace("<br/>", "\n").replace("<br>", "\n").replace("<br />", "\n")
        txt = BeautifulSoup(html_str, "lxml").get_text("\n")
        description = [s.strip() for s in txt.split("\n") if _clean_text(s)]

    # Transaction type
    transaction_type = _detect_transaction_type(title, description)

    # Description price (useful for rent)
    desc_price, desc_curr = _parse_desc_price(description)

    # Choose price: if rent & desc price exists -> prefer desc price
    if transaction_type == "rent" and desc_price is not None:
        price, currency = desc_price, (desc_curr or currency_big)
    else:
        price, currency = price_big, currency_big

    # Area, rooms, bathrooms (adDetailFeature)
    living_area = None
    room_count = None
    bathroom_count = None
    for feat in soup.select(".adDetails .adDetailFeature span"):
        t = _clean_text(feat.get_text())
        if not t:
            continue
        if re.search(r"\bm²\b", t):
            living_area = _to_int(t)
        elif re.search(r"\b(\d+)\s*Rooms?\b", t, re.I):
            room_count = _to_int(t)
        elif re.search(r"\b(\d+)\s*Bathrooms?\b", t, re.I):
            bathroom_count = _to_int(t)

    # General characteristics (outside surface, condition, age, property type)
    land_area = None
    condition = None
    property_type = None
    for box in soup.select(".adMainFeature .adMainFeatureContent"):
        label = _clean_text((box.select_one(".adMainFeatureContentLabel") or {}).get_text() if box.select_one(".adMainFeatureContentLabel") else None)
        value = _clean_text((box.select_one(".adMainFeatureContentValue") or {}).get_text() if box.select_one(".adMainFeatureContentValue") else None)
        if not label or not value:
            continue
        if re.search(r"type of property", label, re.I):
            property_type = value
        elif re.search(r"outside surface", label, re.I):
            land_area = _to_int(value)
        elif re.search(r"condition", label, re.I):
            condition = value
        # age exists but we won't guess a construction year from it

    # Address split
    address, governorate, delegation, locality = _extract_address(soup)

    # Features → booleans (from icon list + description text)
    feature_labels = _collect_feature_labels(soup)
    blob = " ".join(description + feature_labels).lower()

    has_garden = _bool_from_names(feature_labels, "Garden") or (True if "garden" in blob else None)
    has_terrace = _bool_from_names(feature_labels, "Terrace") or (True if "terrace" in blob else None)
    has_garage = _bool_from_names(feature_labels, "Garage")
    has_pool = _bool_from_names(feature_labels, "Pool", "Piscine") or (True if re.search(r"\bpool|piscine\b", blob) else None)
    has_balcony = True if re.search(r"\bbalcony|balcon\b", blob) else None

    air_conditioning = _bool_from_names(feature_labels, "Air conditioning")
    heating = _bool_from_names(feature_labels, "Heating")
    furnished = True if re.search(r"\bfurnished|meublé|meuble\b", blob) else None

    # Phones (usually hidden behind AJAX; still try to sniff any visible numbers)
    page_text = soup.get_text(" ")
    phones = sorted(set(m.group(0).strip() for m in PHONE_RE.finditer(page_text)))

    # Agency / business name
    agency = None
    biz = soup.select_one(".agency-info .businessInfo .businessName a")
    if biz:
        agency = _clean_text(biz.get_text())

    # Photos
    photos = _photos_from_gallery_json(soup) or _photos_from_imgs(soup)
    photos = list(dict.fromkeys(photos)) or None

    # Listing date: not reliably exposed; keep None unless found in meta/time
    listing_date = None
    time_meta = soup.select_one("meta[itemprop='datePublished'][content]") or soup.select_one("time[itemprop='datePublished'][datetime]")
    if time_meta:
        iso = time_meta.get("content") or time_meta.get("datetime")
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            listing_date = int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except Exception:
            listing_date = None

    out = {
        "url": url,
        "id": adid,
        "source": "mubawab",
        "property_type": property_type,
        "price": price,
        "currency": currency,
        "description": description,
        "address": address,
        "governorate": governorate,        # intentionally None unless clearly on page
        "delegation": delegation,
        "locality": locality,
        "postal_code": None,
        "living_area": living_area,
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
        "listing_date": listing_date,
        "last_updated": None,
        "photos": photos,
        "features": feature_labels or None,
        "condition": condition,
        "transaction_type": transaction_type,
    }
    return out

if __name__ == "__main__":
    test_url = "https://www.mubawab.tn/en/a/8124624/great-apartment-for-rent-in-sidi-daoued-3-rooms-double-glazed-windows-and-reinforced-door"
    test_url = "https://www.mubawab.tn/en/a/8146730/find-your-house-to-buy-in-fouchana-small-area-110-m%C2%B2-private-garden"
    data = scrape_mubawab(test_url)
    print(json.dumps(data, ensure_ascii=False, indent=2))
