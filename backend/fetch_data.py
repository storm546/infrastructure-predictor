"""
fetch_data.py - Download REAL Bulgarian procurement data (OCDS format).

Source: Open Contracting Data Registry - Bulgaria (DIGIWHIST / opentender.eu)
  https://data.open-contracting.org/en/publication/44
  License: CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike)

Each yearly file is OCDS JSON-Lines (gzip). Unlike the previous pipeline,
the contract execution period (tender.contractPeriod.durationInDays) is a
REAL field published by the contracting authority - it is not synthesised.
"""

import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ocds"

PUBLICATION = "44"  # Bulgaria
BASE = f"https://data.open-contracting.org/en/publication/{PUBLICATION}/download"

# More years = more data; 2020-2023 gives ~12.6k construction contracts
# with a real duration target.
YEARS = ["2020", "2021", "2022", "2023"]


def download_year(year: str) -> Path:
    """Download one yearly OCDS JSON-Lines gzip file."""
    out = RAW_DIR / f"{year}.jsonl.gz"
    if out.exists() and out.stat().st_size > 0:
        print(f"  [{year}] already present ({out.stat().st_size:,} bytes)")
        return out

    url = f"{BASE}?name={year}.jsonl.gz"
    print(f"  [{year}] downloading {url}")
    resp = requests.get(
        url, timeout=300, headers={"User-Agent": "infrastructure-predictor/2.0"}
    )
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print(f"  [{year}] saved {len(resp.content):,} bytes -> {out}")
    return out


def main():
    print("=" * 60)
    print("  Bulgarian Procurement Data Fetcher (OCDS / Open Contracting)")
    print("=" * 60)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for year in YEARS:
        try:
            download_year(year)
            ok += 1
        except Exception as e:
            print(f"  [{year}] ERROR: {e}", file=sys.stderr)

    print(f"\n  Downloaded {ok}/{len(YEARS)} yearly files into {RAW_DIR}")
    if ok == 0:
        print("  No data fetched. Check network to data.open-contracting.org")
        sys.exit(1)


if __name__ == "__main__":
    main()
