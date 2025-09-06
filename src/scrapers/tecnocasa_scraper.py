import re
import json
import html as ihtml
import requests
from bs4 import BeautifulSoup
from datetime import datetime

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

def _clean_text(t):
    if not t:
        return None
    t = re.sub(r"\s+", " ", t).strip()
    return t or None

def _only_digits(s):
    if not s:
        return None
    digits = re.sub(r"[^\d]", "", str(s))
    return digits or None

def _to_int(s):
    d = _only_digits(s)
    return int(d) if d else None

def _epoch_ms(dt_str):
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(dt_str, fmt).timestamp() * 1000)
        except ValueError:
            continue
    return None

def _parse_description(soup, estate_json):
    # Prefer JSON (already HTML-escaped inside); fallback to slot template
    if estate_json and estate_json.get("description"):
        try:
            desc_html = ihtml.unescape(estate_json["description"])
            desc_soup = BeautifulSoup(desc_html, "html.parser")
            paras = [ _clean_text(p.get_text(" ", strip=True)) for p in desc_soup.select("p") ]
            return [p for p in paras if p]
        except Exception:
            pass

    slot = soup.select_one('template[slot="estate-description"]')
    if slot:
        paras = [ _clean_text(p.get_text(" ", strip=True)) for p in slot.select("p") ]
        out = [p for p in paras if p]
        if out:
            return out
    return None

def _parse_price_currency_from_json(estate_json):
    if not estate_json:
        return None, "TND"
    if estate_json.get("private_negotiation"):
        return None, "TND"
    # 1) numeric_price if present
    if "numeric_price" in estate_json and estate_json["numeric_price"]:
        return str(int(estate_json["numeric_price"])), "TND"
    # 2) textual price
    price_txt = estate_json.get("price") or (estate_json.get("costs") or {}).get("price")
    if price_txt:
        digits = _only_digits(price_txt)
        cur = "EUR" if "€" in price_txt or "EUR" in price_txt.upper() else "TND"
        # DT / Dinar → TND
        if re.search(r"\bDT\b|\bDINAR", price_txt, re.I):
            cur = "TND"
        if digits:
            return digits, cur
    return None, "TND"

def _parse_price_currency_from_dom(soup, html_text):
    # Look inside the "Frais > Prix" block first
    for row in soup.select('template[slot="estate-costs"] .row, .costs-container .row'):
        txt = _clean_text(row.get_text(" ", strip=True))
        if txt and "prix" in txt.lower():
            # e.g. "Prix: 400 000 DT"
            m = re.search(r"([\d\s\.,]{3,})\s*(DT|TND|EUR|€)?", txt, re.I)
            if m:
                digits = _only_digits(m.group(1))
                cur = (m.group(2) or "").upper().replace("€", "EUR")
                if cur in ("", None, "DT"): cur = "TND"
                return digits, cur or "TND"

    # Classic price spot (some pages also render this)
    el = soup.select_one(".estate-price .current-price")
    if el:
        txt = _clean_text(el.get_text(" ", strip=True))
        m = re.search(r"([\d\s\.,]{3,})\s*(DT|TND|EUR|€)?", txt or "", re.I)
        if m:
            digits = _only_digits(m.group(1))
            cur = (m.group(2) or "").upper().replace("€", "EUR")
            if cur in ("", None, "DT"): cur = "TND"
            return digits, cur or "TND"

    # Last resort: sweep whole HTML
    m = re.search(r"([\d\s\.,]{3,})\s*(DT|TND|EUR|€)\b", html_text, re.I)
    if m:
        digits = _only_digits(m.group(1))
        cur = m.group(2).upper().replace("€", "EUR")
        if cur == "DT": cur = "TND"
        return digits, cur
    return None, "TND"

def _get_from_data_list(estate_json, label):
    """
    estate_json["data"] is a list of {label, valore}. Labels are localized.
    """
    if not estate_json or not estate_json.get("data"):
        return None
    for item in estate_json["data"]:
        if _clean_text(item.get("label", "")).lower() == label.lower():
            return _clean_text(str(item.get("valore")))
    return None

def _parse_floor(estate_json, soup):
    v = None
    # JSON: features.floor or data list label 'étage' (varies case/accents)
    if estate_json:
        v = (estate_json.get("features") or {}).get("floor") or _get_from_data_list(estate_json, "étage")
    if not v:
        # DOM fallback:
        for row in soup.select('template[slot="estate-features"] .row'):
            rowtxt = _clean_text(row.get_text(" ", strip=True))
            if not rowtxt: 
                continue
            if re.search(r"\b(etage|étage)\b", rowtxt, re.I):
                m = re.search(r"(\d+)", rowtxt)
                if m:
                    v = m.group(1)
                    break
    return _to_int(v) if v else None

def _parse_photos(estate_json):
    urls = []
    if estate_json and estate_json.get("media") and estate_json["media"].get("images"):
        for it in estate_json["media"]["images"]:
            u = (
                (it.get("url") or {}).get("detail") or
                (it.get("url") or {}).get("gallery") or
                (it.get("url") or {}).get("card")
            )
            if u:
                urls.append(u)
    # unique & keep order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def _parse_booleans_from_desc(desc_list):
    """Heuristics for balcony/terrace/garden/pool/garage from description text."""
    text = " ".join(desc_list).lower() if desc_list else ""
    def has(word):
        return bool(re.search(rf"\b{word}\w*\b", text))
    return {
        "has_balcony": has("balcon"),
        "has_terrace": has("terrass"),
        "has_garden": has("jardin"),
        "has_pool": has("piscine"),
        "has_garage": has("garage"),
    }

def scrape_tecnocasa(url: str) -> dict:
    resp = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "fr,en;q=0.8"}, timeout=25)
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # ---------- 1) Pull the big JSON from <bottom-menu :estate="…"> ----------
    estate_json = None
    bm = soup.find("bottom-menu")
    if bm:
        # Attribute can be ':estate' or 'estate' (colon is valid)
        estate_attr = bm.attrs.get(":estate") or bm.attrs.get("estate")
        if estate_attr:
            # Unescape HTML entities twice just in case (&quot; etc.)
            raw = ihtml.unescape(ihtml.unescape(estate_attr))
            # Some libraries may wrap it with stray spaces/newlines
            raw = raw.strip()
            try:
                estate_json = json.loads(raw)
            except json.JSONDecodeError:
                # Very rare: extract the {...} with a greedy brace match
                m = re.search(r"\{.*\}\s*$", raw, re.S)
                if m:
                    estate_json = json.loads(m.group(0))

    # ---------- 2) Core fields ----------
    # id
    prop_id = None
    if estate_json and estate_json.get("id"):
        prop_id = str(estate_json["id"])
    else:
        m = re.search(r"/(\d+)\.html?$", url)
        if m:
            prop_id = m.group(1)

    # property type
    prop_type = None
    if estate_json:
        prop_type = (estate_json.get("type") or {}).get("title") or _get_from_data_list(estate_json, "Typologie")
    if prop_type:
        prop_type = _clean_text(prop_type)

    # address pieces
    governorate = delegation = locality = postal = None
    if estate_json:
        governorate = (estate_json.get("province") or {}).get("title") or _get_from_data_list(estate_json, "Région")
        delegation = (estate_json.get("city") or {}).get("title") or _get_from_data_list(estate_json, "Municipalité")
        postal = _get_from_data_list(estate_json, "Postal Code") or (estate_json.get("agency") or {}).get("zip_code")

    # Compose address: "<Delegation>, <Governorate>" when both, else available parts
    address_parts = [p for p in [delegation, governorate] if p]
    address = ", ".join(address_parts) if address_parts else None

    # living / land area
    living_area = None
    if estate_json:
        living_area = estate_json.get("numeric_surface") or _get_from_data_list(estate_json, "Surface") or estate_json.get("surface")
    if living_area:
        living_area = _only_digits(living_area)

    land_area = None  # not provided on apartments; villa pages sometimes have a different field

    # rooms (Pièces) — number of rooms, not bedrooms
    room_count = None
    if estate_json:
        room_count = _get_from_data_list(estate_json, "Pièces") or estate_json.get("rooms")
    if isinstance(room_count, str):
        m = re.search(r"\d+", room_count)
        room_count = int(m.group(0)) if m else None
    elif isinstance(room_count, (int, float)):
        room_count = int(room_count) if room_count is not None else None

    # floor
    floor = _parse_floor(estate_json, soup)

    # year
    construction_year = None
    if estate_json:
        construction_year = (estate_json.get("dates") or {}).get("build_year") or (estate_json.get("features") or {}).get("build_year")
    construction_year = _to_int(construction_year) if construction_year else None

    # description paragraphs
    description = _parse_description(soup, estate_json)

    # price + currency (JSON first, then DOM)
    price, currency = _parse_price_currency_from_json(estate_json)
    if not price:
        price, currency = _parse_price_currency_from_dom(soup, html)
    # Normalize currency defaults
    if currency == "DT":
        currency = "TND"
    if currency not in ("TND", "EUR"):
        currency = "TND"

    # photos
    photos = _parse_photos(estate_json)

    # heating / AC / furnished (best-effort)
    heating = air_conditioning = furnished = None
    if estate_json:
        f = estate_json.get("features") or {}
        heating = f.get("heating") or (estate_json.get("energy_data") or {}).get("heating")
        air_conditioning = f.get("air_conditioning")
        furnished = f.get("furnitured")

    # Booleans from description (balcony/terrace/garden/pool/garage)
    bools = _parse_booleans_from_desc(description)

    # agency, phones
    agency = contact_name = None
    phones = []
    if estate_json:
        ag = estate_json.get("agency") or {}
        agency = ag.get("name")
        for key in ("phone", "whatsapp", "phone_real", "phone_proxy"):
            v = ag.get(key)
            if v:
                digits = re.sub(r"[^\d\+]", "", str(v))
                if digits and digits not in phones:
                    phones.append(digits)

    # listing date
    listing_date = None
    if estate_json:
        listing_date = _epoch_ms(estate_json.get("last_published_at"))

    # transaction type
    transaction_type = "sale"
    if estate_json:
        c = (estate_json.get("contract") or {}).get("slug") or (estate_json.get("contract") or {}).get("slug_estate")
        if c and "louer" in c:
            transaction_type = "rent"

    # Final payload
    result = {
        "url": url,
        "id": prop_id,
        "source": "tecnocasa",
        "property_type": prop_type,
        "price": price,
        "currency": currency or "TND",
        "description": description,
        "address": address,
        "governorate": _clean_text(governorate),
        "delegation": _clean_text(delegation),
        "locality": _clean_text(locality),
        "postal_code": _clean_text(postal),
        "living_area": living_area,
        "land_area": land_area,
        "room_count": room_count,
        "bathroom_count": None,  # not exposed on this site; keep None
        "construction_year": construction_year,
        "floor": floor,
        "has_garage": bools["has_garage"] if bools else None,
        "has_garden": bools["has_garden"] if bools else None,
        "has_pool": bools["has_pool"] if bools else None,
        "has_balcony": bools["has_balcony"] if bools else None,
        "has_terrace": bools["has_terrace"] if bools else None,
        "heating": _clean_text(heating),
        "air_conditioning": _clean_text(air_conditioning),
        "furnished": _clean_text(furnished),
        "phone": phones or None,
        "agency": _clean_text(agency),
        "contact_name": _clean_text(contact_name),
        "listing_date": listing_date,
        "last_updated": None,
        "photos": photos,
        "features": None,     # optional: you can pack estate_json["data"] here if you want
        "condition": None,
        "transaction_type": transaction_type,
    }
    return result

# --- quick sanity check (example) ---
# print(json.dumps(scrape_tecnocasa("https://www.tecnocasa.tn/vendre/appartement/bizerte/bizerte/57418.html"), indent=2, ensure_ascii=False))
