"""
fetch_data.py - Real Bulgarian Public Procurement Data Acquisition

Sources:
  1. data.egov.bg - Public Procurement Agency (AOP) contract datasets
  2. Stara Zagora municipality announcements (attempted)
  3. CAIS EOP (requires auth - public datasets used instead)

Downloads multiple years of contract data + amendments from the Bulgarian
Open Data Portal. Contract dates serve as start_dates; amendment data
is used to derive completion timelines.
"""

import requests
import zipfile
import io
import os
import sys
import glob
import csv
from pathlib import Path

# Dataset UUIDs from data.egov.bg
# Format: {year: dataset_uuid}
DATASETS = {
    "2024": "88ea1672-944b-4b9a-b074-528e316eab46",
    "2025": "7990cb41-719d-4616-b656-c750ebb487d7",
}

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"


def download_dataset_zip(year: str, uuid: str) -> bytes:
    """Download all resources as a CSV ZIP from the open data portal."""
    # Step 1: GET the JSON redirect info
    json_url = f"https://data.egov.bg/dataset/{uuid}/resources/download/csv"
    resp = requests.get(json_url, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    # Step 2: Follow the redirect to the actual ZIP
    zip_url = (
        f"https://data.egov.bg/dataset/resources/download/zip/"
        f"{meta['format']}/{meta['uri']}/{meta['delete_only_zip']}"
    )
    resp = requests.get(zip_url, timeout=60)
    resp.raise_for_status()

    if not resp.content[:2] == b"PK":
        raise RuntimeError(f"Downloaded content is not a ZIP file for {year}")

    return resp.content


def extract_contracts(zip_bytes: bytes, year: str) -> list[dict]:
    """Extract contracts CSV from ZIP and return list of dicts."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # Find the contracts file (not annexes)
        contracts_name = None
        for name in zf.namelist():
            if "contracts" in name.lower() and "annex" not in name.lower():
                contracts_name = name
                break

        if not contracts_name:
            raise RuntimeError(f"No contracts CSV found in ZIP for {year}")

        with zf.open(contracts_name) as f:
            text = f.read().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)

    # Add year column
    for row in rows:
        row["DATA_YEAR"] = year
        row["SOURCE_FILE"] = contracts_name

    return rows


def extract_annexes(zip_bytes: bytes, year: str) -> list[dict]:
    """Extract annexes CSV from ZIP."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        annexes_name = None
        for name in zf.namelist():
            if "annex" in name.lower() or "izmeneniya" in name.lower():
                annexes_name = name
                break

        if not annexes_name:
            return []

        with zf.open(annexes_name) as f:
            text = f.read().decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)

    for row in rows:
        row["DATA_YEAR"] = year

    return rows


def save_csv(rows: list[dict], filename: str):
    """Save list of dicts to CSV."""
    if not rows:
        print(f"  No data to save for {filename}")
        return

    filepath = RAW_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    keys = rows[0].keys()
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} rows -> {filepath}")


def main():
    print("=" * 60)
    print("  Bulgarian Public Procurement Data Fetcher")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_contracts = []
    all_annexes = []

    for year, uuid in DATASETS.items():
        print(f"\n[{year}] Downloading from data.egov.bg...")
        try:
            zip_bytes = download_dataset_zip(year, uuid)
            print(f"  Downloaded {len(zip_bytes):,} bytes")

            contracts = extract_contracts(zip_bytes, year)
            print(f"  Extracted {len(contracts)} contracts")

            annexes = extract_annexes(zip_bytes, year)
            print(f"  Extracted {len(annexes)} annexes")

            all_contracts.extend(contracts)
            all_annexes.extend(annexes)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save raw CSVs
    save_csv(all_contracts, "contracts_raw.csv")
    save_csv(all_annexes, "annexes_raw.csv")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {len(all_contracts)} contracts, {len(all_annexes)} annexes")
    print(f"  Data saved to {RAW_DIR}")
    print(f"{'=' * 60}")

    return all_contracts, all_annexes


if __name__ == "__main__":
    main()
