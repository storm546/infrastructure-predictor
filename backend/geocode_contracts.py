"""
geocode_contracts.py - Extract street-level locations and geocode via Nominatim.
"""
import re
import json
import time
import requests
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "processed" / "geocode_cache.json"
CLEAN_PATH = DATA_DIR / "processed" / "contracts_clean.csv"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "InfrastructurePredictor/1.0 (stormlabs.cloud)"}

# Bulgarian quote characters to strip
BULGARIAN_QUOTES = '\u201e\u201c\u201d\u00ab\u00bb"'


def strip_quotes(s: str) -> str:
    for q in BULGARIAN_QUOTES:
        s = s.replace(q, '')
    return s.strip()


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def extract_location_string(subject: str, town: str) -> str:
    """Build a clean geocoding query from contract subject + town."""
    if not subject or pd.isna(subject):
        return town if town != "unknown" else "Bulgaria"

    s = str(subject)
    tc = town if town != "unknown" else ""

    # Clean quotes first for simpler matching
    for q in BULGARIAN_QUOTES:
        s = s.replace(q, ' ')

    # ul. X or bul. X - capture street name, stop at delimiters
    m = re.search(r'(?:ул\.|бул\.)\s+([^,]+?)(?:\s+(?:от|до)\s|,\s*|$)', s)
    if m:
        name = m.group(1).strip()
        # Take only first 1-2 words of the street name
        words = name.split()
        if len(words) > 2:
            # Check if first word is a single letter (like "Д-р")
            if len(words[0]) <= 3:
                name = ' '.join(words[:3])
            else:
                name = ' '.join(words[:2])
        if len(name) > 2:
            return f"{name} {tc}" if tc else name

    # zh.k. X
    m = re.search(r'ж\.\s*к\.\s+(\S+(?:\s+\S+){0,1})', s)
    if m:
        name = m.group(1).strip()
        if len(name) > 1:
            return f"{name} {tc}" if tc else name

    # s. X (village)
    m = re.search(r'с\.\s+(\S+(?:\s+\S+){0,1})', s)
    if m and len(m.group(1)) > 2:
        return f"{m.group(1)} Bulgaria"

    # raion X
    m = re.search(r'район\s+(\S+)', s)
    if m and len(m.group(1)) > 2:
        return f"{m.group(1)} {tc}" if tc else m.group(1)

    # pat VARXXXX
    m = re.search(r'път\s+(VAR\s*\d+)', s)
    if m:
        road = re.sub(r'\s+', '', m.group(1))
        return f"pat {road} {tc}" if tc else f"pat {road}"

    return f"{tc} Bulgaria" if tc else "Bulgaria"


def geocode(query: str) -> tuple | None:
    """Geocode a query string using Nominatim."""
    try:
        params = {"q": query, "format": "json", "limit": 1}
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"  Geocode error for '{query[:80]}': {e}")
    return None


def main():
    cache = load_cache()
    print(f"Loaded {len(cache)} cached geocodes")

    if not CLEAN_PATH.exists():
        print("No clean data. Run clean_data.py first.")
        return

    df = pd.read_csv(CLEAN_PATH)
    new = 0

    for i, row in df.iterrows():
        subject = str(row.get("ПРЕДМЕТ на договора", ""))
        town = str(row.get("town", ""))
        cn = str(row.get("ДОГОВОР НОМЕР", ""))

        if cn in cache:
            continue

        q = extract_location_string(subject, town)
        print(f"[{i+1}/{len(df)}] {cn}: {q[:100]}")
        coords = geocode(q)

        if coords:
            cache[cn] = {"lat": coords[0], "lng": coords[1], "query": q}
            new += 1
        else:
            cache[cn] = None

        time.sleep(1.1)

    save_cache(cache)
    print(f"\nDone: {new} new geocodes, {len(cache)} total")


if __name__ == "__main__":
    main()
