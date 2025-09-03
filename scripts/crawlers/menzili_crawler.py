# pip install requests beautifulsoup4 lxml
import json
import os
import time
import random
from typing import List, Dict, Any, Set
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

import requests
from bs4 import BeautifulSoup

# If your scrape_menzili is in another module, import it instead:
# from your_module import scrape_menzili
try:
    from src.scrapers.menzili_scraper import scrape_menzili
except ModuleNotFoundError:
    # Fallback: adjust the import if running as a script or from another location
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.scrapers.menzili_scraper import scrape_menzili
# def scrape_menzili(url: str) -> Dict[str, Any]:
#     """
#     Placeholder. Replace by your real implementation that returns a JSON-able dict.
#     Must at least include the "url" key in the returned dict.
#     """
#     raise NotImplementedError("Import your real scrape_menzili(url) function here.")


LISTING_START_URL = "https://www.menzili.tn/immo/vente-maison-tunisie?page=1&tri=1"
OUTPUT_JSON_PATH = r"C:\Users\Razer\Documents\2025_work\House_price_prediction\data\raw\menzili_listings.json"
PAGES_TO_SCAN = 3
REQUEST_TIMEOUT = (10, 25)  # (connect, read)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "fr,en;q=0.9",
}

def set_query_param(url: str, **params) -> str:
    """
    Returns a copy of `url` with the given query params set/replaced.
    """
    parts = list(urlparse(url))
    q = parse_qs(parts[4], keep_blank_values=True)
    for k, v in params.items():
        q[str(k)] = [str(v)]
    parts[4] = urlencode(q, doseq=True)
    return urlunparse(parts)

def normalize_url(u: str) -> str:
    """
    Normalize URL for comparison: lower host, remove fragments and trailing slash.
    """
    p = urlparse(u)
    # Force https scheme and lower netloc
    scheme = "https" if p.scheme in ("http", "https") else "https"
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))

def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text

def extract_post_urls_from_listing_html(html: str) -> List[str]:
    """
    Extract menzili post URLs from listing pages.
    Strategy:
      1) anchors with class 'li-item-list-title'
      2) any anchor under '.li-item-list' whose href contains '/annonce/'
    """
    soup = BeautifulSoup(html, "lxml")
    urls: List[str] = []

    # Primary: the title link
    for a in soup.select("div.li-item-list a.li-item-list-title[href]"):
        href = a.get("href")
        if href:
            urls.append(href)

    # Secondary: any detail link within the item block
    for a in soup.select("div.li-item-list a[href*='/annonce/']"):
        href = a.get("href")
        if href:
            urls.append(href)

    # Dedupe while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for u in urls:
        nu = normalize_url(u)
        if nu not in seen:
            seen.add(nu)
            unique.append(u)
    return unique

def load_existing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # if file accidentally contains a dict, wrap into list
            return [data]
        except json.JSONDecodeError:
            # Corrupt file: back it up and start fresh
            os.rename(path, path + ".bak")
            return []

def save_all(path: str, data: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def get_listing_page_urls(start_url: str, pages: int) -> List[str]:
    """
    Build the URLs for pages 1..pages, preserving tri=1 (sort by date).
    """
    out = []
    for p in range(1, pages + 1):
        out.append(set_query_param(start_url, page=p, tri=1))
    return out

def crawl_menzili_latest(
    start_url: str = LISTING_START_URL,
    pages_to_scan: int = PAGES_TO_SCAN,
    out_json_path: str = OUTPUT_JSON_PATH,
    polite_sleep_range=(1.0, 2.0),
) -> Dict[str, Any]:
    """
    - If out_json_path exists: skip posts whose normalized URL already exists in the saved list.
    - If it doesn't exist: scrape ALL posts found on the first `pages_to_scan` listing pages.
    Returns a small summary with counts.
    """
    existing_items = load_existing(out_json_path)
    existing_by_url: Set[str] = set()
    if existing_items:
        for it in existing_items:
            u = it.get("url")
            if u:
                existing_by_url.add(normalize_url(u))

    pages = get_listing_page_urls(start_url, pages_to_scan)
    all_listing_urls: List[str] = []
    for page_url in pages:
        try:
            html = fetch(page_url)
        except Exception as e:
            print(f"[WARN] Failed to fetch listing page {page_url}: {e}")
            continue
        page_urls = extract_post_urls_from_listing_html(html)
        all_listing_urls.extend(page_urls)
        # be a tiny bit polite between listing pages
        time.sleep(random.uniform(*polite_sleep_range))

    # Deduplicate post URLs across pages (preserve order)
    seen_norm: Set[str] = set()
    dedup_listing_urls: List[str] = []
    for u in all_listing_urls:
        nu = normalize_url(u)
        if nu not in seen_norm:
            seen_norm.add(nu)
            dedup_listing_urls.append(u)

    to_scrape: List[str] = []
    if existing_items:
        # Compare only if file exists
        for u in dedup_listing_urls:
            nu = normalize_url(u)
            if nu not in existing_by_url:
                to_scrape.append(u)
    else:
        # No file -> scrape everything (no comparing)
        to_scrape = dedup_listing_urls[:]

    print(f"[INFO] Found {len(dedup_listing_urls)} listing URLs; scraping {len(to_scrape)} new.")

    new_items: List[Dict[str, Any]] = []
    for i, post_url in enumerate(to_scrape, 1):
        try:
            item = scrape_menzili(post_url)
            if isinstance(item, dict):
                # Ensure the 'url' field is present for future comparisons
                item.setdefault("url", post_url)
                new_items.append(item)
                print(f"[OK] ({i}/{len(to_scrape)}) {post_url}")
            else:
                print(f"[WARN] scrape_menzili returned non-dict for {post_url}, skipping.")
        except Exception as e:
            print(f"[ERR] Failed to scrape {post_url}: {e}")
        # politeness between post fetches
        time.sleep(random.uniform(*polite_sleep_range))

    # Merge and save
    if existing_items:
        merged = existing_items + new_items
    else:
        merged = new_items

    save_all(out_json_path, merged)

    return {
        "pages_scanned": len(pages),
        "found_urls": len(dedup_listing_urls),
        "scraped_new": len(new_items),
        "total_saved": len(merged),
        "output_file": out_json_path,
    }


if __name__ == "__main__":
    summary = crawl_menzili_latest()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
