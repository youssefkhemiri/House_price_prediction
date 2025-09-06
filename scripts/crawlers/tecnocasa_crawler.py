# tecnocasa_crawler.py
import os

# --- import your single-URL scraper (adjust module name if needed) ---
try:
    from src.scrapers.tecnocasa_scraper import scrape_tecnocasa 
except ModuleNotFoundError:
    # Fallback: adjust the import if running as a script or from another location
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from src.scrapers.tecnocasa_scraper import scrape_tecnocasa

# tecnocasa_selenium_crawler.py
import re
import json
import time
import random
from typing import List, Dict, Set, Optional

# 👉 your single-listing scraper (old version) — keep this import name as you asked
#    if your module/file name is different, adjust just this line.

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE = "https://www.tecnocasa.tn"
MACRO = "vendre/immeubles/nord-est-ne"
BASE_LIST = f"{BASE}/{MACRO}/"

# regions in Nord-Est (as shown in the website’s filter)
REGIONS = [
    "bizerte",
    "cap-bon",
    "grand-tunis",
    "kairouan",
    "mahdia",
    "monastir",
    "sfax",
    "sousse",
]

MAX_PAGES_PER_MUNICIPALITY = 5


# ---------- Selenium setup ----------

def _make_driver() -> webdriver.Chrome:
    opts = Options()
    # comment next line if you want to see the browser
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1280,1800")
    opts.add_argument("--lang=fr-FR,fr")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    # small anti-bot hardening
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    # make loads faster; DOMContentLoaded is enough for our parsing
    driver.set_page_load_timeout(40)
    return driver


def _sleep(a=0.7, b=1.6):
    time.sleep(random.uniform(a, b))


# ---------- discovery helpers ----------

def _region_url(region_slug: str) -> str:
    return f"{BASE_LIST}{region_slug}.html"


def _municipality_url(region_slug: str, municipality_slug: str, page: int) -> str:
    base_url = f"{BASE_LIST}{region_slug}/{municipality_slug}.html"
    if page <= 1:
        return base_url
    return base_url + f"/pag-{page}"


def _discover_municipalities_from_html(html: str, region_slug: str) -> List[str]:
    """
    Find municipality slugs inside a region page.
    Looks for anchors like:
      /vendre/immeubles/nord-est-ne/{region}/{municipality}.html
    """
    pat = re.compile(
        rf"/{MACRO}/{re.escape(region_slug)}/([a-z0-9\-]+)\.html",
        re.I,
    )
    munis = sorted(set(pat.findall(html)))
    return munis


def _extract_detail_links_from_html(html: str) -> List[str]:
    """
    Extract detail listing URLs:
      https://www.tecnocasa.tn/vendre/<type>/<region>/<municipality>/<id>.html
    """
    pat = re.compile(
        r"https?://www\.tecnocasa\.tn/vendre/[a-z0-9\-]+/[a-z0-9\-]+/[a-z0-9\-]+/\d+\.html",
        re.I,
    )
    return sorted(set(m.group(0) for m in pat.finditer(html)))


def _try_close_cookie_banner(driver: webdriver.Chrome):
    """Try to close any cookie/overlay that might block content; best-effort."""
    try:
        # common patterns; we don't block if we fail
        # buttons with text like "Accepter", "OK", "J'accepte", etc.
        for text in ["Accepter", "J'accepte", "OK", "Ok", "D'accord"]:
            els = driver.find_elements(By.XPATH, f"//button[contains(., '{text}')]")
            if els:
                els[0].click()
                _sleep(0.4, 0.9)
                break
    except Exception:
        pass


# ---------- main crawler ----------

def crawl_tecnocasa_selenium() -> List[Dict]:
    """
    1) For each region, discover municipalities dynamically.
    2) For each municipality, open first 5 pages and collect listing links.
    3) For each listing link, call scrap_tecnocasa(url) to get JSON.
    Returns a list of JSON rows.
    """
    driver = _make_driver()
    all_results: List[Dict] = []

    try:
        # Pre-warm site (cookies, etc.)
        driver.get(BASE + "/")
        _try_close_cookie_banner(driver)
        _sleep()

        # ---------- 1) discover municipalities ----------
        region_to_munis: Dict[str, List[str]] = {}
        for region in REGIONS:
            url = _region_url(region)
            print(f"[region] {url}")
            driver.get(url)
            _try_close_cookie_banner(driver)
            # let content paint
            _sleep(0.8, 1.8)
            html = driver.page_source
            munis = _discover_municipalities_from_html(html, region)

            # If nothing found, try also scanning any “districts-seo-list” links on the page
            if not munis:
                extra = re.findall(
                    rf'href="(/{MACRO}/{re.escape(region)}/[a-z0-9\-]+\.html)"',
                    html, flags=re.I
                )
                munis = sorted(set(
                    re.search(r"/([a-z0-9\-]+)\.html$", path).group(1)
                    for path in extra if re.search(r"/([a-z0-9\-]+)\.html$", path)
                ))

            if munis:
                print(f"  -> {len(munis)} municipalities: {', '.join(munis[:8])}{'…' if len(munis) > 8 else ''}")
            else:
                print(f"  -> no municipalities found (could be bot-guard). You can hardcode a few if needed.")
            region_to_munis[region] = munis
            _sleep()

        # ---------- 2) collect detail links ----------
        all_detail_links: Set[str] = set()
        for region, munis in region_to_munis.items():
            for muni in munis:
                print(f"[municipality] {region}/{muni}")
                for page in range(1, MAX_PAGES_PER_MUNICIPALITY + 1):
                    list_url = _municipality_url(region, muni, page)
                    print(f"  [page {page}] {list_url}")
                    driver.get(list_url)

                    # wait up to 12s for any estate card to appear OR pagination (page exists)
                    try:
                        WebDriverWait(driver, 12).until(
                            EC.any_of(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.estate-card a")),
                                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.pagination")),
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.estates-list"))
                            )
                        )
                    except Exception:
                        # even if wait fails, we still parse page_source; sometimes enough
                        pass

                    _sleep(0.4, 1.1)
                    html = driver.page_source

                    # take links from DOM anchors (more reliable than regex alone)
                    dom_links = [a.get_attribute("href")
                                 for a in driver.find_elements(By.CSS_SELECTOR, "div.estate-card a[href]")]
                    dom_links = [u for u in dom_links if u and "/vendre/" in u and u.endswith(".html")]
                    regex_links = _extract_detail_links_from_html(html)

                    found = sorted(set(dom_links) | set(regex_links))
                    print(f"    -> found {len(found)} detail links on page {page}")

                    if not found and page > 1:
                        print("    -> stopping early for this municipality (no results)")
                        break

                    before = len(all_detail_links)
                    all_detail_links.update(found)
                    gained = len(all_detail_links) - before
                    if gained:
                        print(f"    -> total unique so far: {len(all_detail_links)} (+{gained})")

                    _sleep()

        print(f"[total detail links collected] {len(all_detail_links)}")

        # Save the discovered URLs to a file
        
        try:
            urls_list = sorted(all_detail_links)
            with open("tecnocasa_urls.json", "w", encoding="utf-8") as f:
                json.dump(urls_list, f, ensure_ascii=False, indent=2)
            
            with open("tecnocasa_urls.txt", "w", encoding="utf-8") as f:
                for url in urls_list:
                    f.write(url + "\n")
        except Exception as e:
            print(f"   ! error saving URLs to file: {e}")
        
        print(f"[URLs saved] {len(urls_list)} URLs saved to tecnocasa_urls.json and tecnocasa_urls.txt")

        # ---------- 3) scrape each detail with your function ----------
        for i, link in enumerate(sorted(all_detail_links), 1):
            try:
                print(f"[{i}/{len(all_detail_links)}] scraping {link}")
                row = scrape_tecnocasa(link)
                all_results.append(row)
            except Exception as e:
                print(f"   ! error scraping {link}: {e}")
            _sleep(0.5, 1.2)

        return all_results

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ---------- CLI ----------
if __name__ == "__main__":
    rows = crawl_tecnocasa_selenium()
    with open("tecnocasa_listings.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open("tecnocasa_listings.ndjson", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved {len(rows)} records -> tecnocasa_listings.json / tecnocasa_listings.ndjson")
    print("URLs list saved -> tecnocasa_urls.json / tecnocasa_urls.txt")
