#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Crawl Mubawab TN newest listings (pages 1..3), dedupe by URL against a saved
JSON *array* file, and for each new URL call your `scrape_mubawab(url)` to
extract a dict which is appended to disk.

Put your `scrape_mubawab(url)` in the same file, or import it:
from your_module import scrape_mubawab
"""

import json
import os
import time
from typing import Iterable, List, Dict, Set, Optional
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag

import requests
from bs4 import BeautifulSoup

# --- If scrape_mubawab is in another module, import it here ---
# from your_module import scrape_mubawab

try:
    from src.scrapers.mubawab_scraper import scrape_mubawab 
except ModuleNotFoundError:
    # Fallback: adjust the import if running as a script or from another location
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.scrapers.mubawab_scraper import scrape_mubawab

# ---------- CONFIG ----------
BASE_URL_TEMPLATE = "https://www.mubawab.tn/en/ct/tunis/real-estate-for-sale:o:n:p:{page}"
PAGES_TO_FETCH = [1, 2, 3]
OUTPUT_JSON_PATH = r"C:\Users\Razer\Documents\2025_work\House_price_prediction\data\raw\mubawab_listings.json"

REQUEST_TIMEOUT = 20  # seconds
SLEEP_BETWEEN_REQUESTS = 1.0  # politeness between page fetches
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,ar;q=0.7",
}
# ----------------------------


def normalize_url(url: str, base: Optional[str] = None) -> Optional[str]:
    """Absolutize, strip fragment, lower-case scheme/host, drop trailing slash."""
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    url, _ = urldefrag(url)
    p = urlparse(url)
    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()
    path = p.path.rstrip("/")
    return urlunparse((scheme, netloc, path, p.params, p.query, ""))


def extract_listing_urls_from_html(html: str, page_url: str) -> List[str]:
    """Grab listing links from the index page."""
    soup = BeautifulSoup(html, "html.parser")
    urls: Set[str] = set()

    # Primary source: listing boxes with linkref
    for box in soup.select("div.listingBox[linkref]"):
        u = normalize_url(box.get("linkref"), base=page_url)
        if u:
            urls.add(u)

    # Also pick anchors under the title (belt-and-suspenders)
    for a in soup.select("h2.listingTit a[href]"):
        u = normalize_url(a.get("href"), base=page_url)
        if u:
            urls.add(u)

    # Conservative fallback
    for a in soup.select("a[href*='/en/a/']"):
        u = normalize_url(a.get("href"), base=page_url)
        if u:
            urls.add(u)

    return sorted(urls)


def load_saved_items(path: str) -> List[Dict]:
    """Load saved JSON array; return [] if not present/invalid."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            return data["items"]
    except Exception:
        pass
    return []


def collect_existing_urls(items: Iterable[Dict]) -> Set[str]:
    """Build a set of known URLs from common keys."""
    keys = ("source_url", "url", "listing_url")
    out: Set[str] = set()
    for it in items:
        for k in keys:
            v = it.get(k)
            if isinstance(v, str) and v:
                out.add(normalize_url(v))
                break
    return out


def save_items(path: str, items: List[Dict]) -> None:
    """Atomic write."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def fetch(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.text


def crawl_index_pages(pages: Iterable[int]) -> List[str]:
    all_urls: Set[str] = set()
    for p in pages:
        page_url = BASE_URL_TEMPLATE.format(page=p)
        try:
            html = fetch(page_url)
        except Exception as e:
            print(f"[WARN] Page {p} fetch failed: {e}")
            continue
        urls = extract_listing_urls_from_html(html, page_url)
        print(f"[INFO] Page {p}: {len(urls)} URLs")
        all_urls.update(urls)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    return sorted(all_urls)


def main():
    # Ensure your function is available
    try:
        scrape_mubawab  # type: ignore  # noqa: F821
    except NameError as e:
        raise RuntimeError(
            "scrape_mubawab(url) is not defined. Import or define it before running."
        ) from e

    saved_items = load_saved_items(OUTPUT_JSON_PATH)
    existing_urls = collect_existing_urls(saved_items)
    print(f"[INFO] Loaded {len(saved_items)} saved items ({len(existing_urls)} URLs)")

    listing_urls = crawl_index_pages(PAGES_TO_FETCH)
    print(f"[INFO] Found {len(listing_urls)} unique listing URLs")

    file_exists = os.path.exists(OUTPUT_JSON_PATH)
    if not file_exists and len(saved_items) == 0:
        to_scrape = listing_urls
        print("[INFO] No saved JSON detected → scraping ALL listings.")
    else:
        to_scrape = [u for u in listing_urls if normalize_url(u) not in existing_urls]
        print(f"[INFO] New URLs to scrape: {len(to_scrape)}")

    appended = 0
    for i, url in enumerate(to_scrape, 1):
        try:
            data = scrape_mubawab(url)  # <-- YOUR EXISTING FUNCTION
            # ensure consistent URL key
            if not data.get("source_url"):
                data["source_url"] = url
            saved_items.append(data)
            appended += 1
            save_items(OUTPUT_JSON_PATH, saved_items)  # incremental persist
            print(f"[OK] [{i}/{len(to_scrape)}] {url}")
        except Exception as e:
            print(f"[ERR] {url}: {e}")
        time.sleep(0.5)  # tiny pause

    save_items(OUTPUT_JSON_PATH, saved_items)
    print(f"[DONE] Added {appended} items. Total saved: {len(saved_items)}")
    print(f"[PATH] {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
