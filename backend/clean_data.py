"""
clean_data.py - Transform raw OCDS procurement data into ML-ready features.

Reads OCDS JSON-Lines (from fetch_data.py) and produces a clean table of
Bulgarian *construction / infrastructure* contracts (CPV division 45).

Key point: the target `contracted_days` is the REAL execution period the
contracting authority published (tender.contractPeriod.durationInDays). It is
the duration agreed at signing - NOT a synthesised/estimated value, and NOT the
actual on-the-ground completion time (open data does not expose that).
"""

import glob
import gzip
import json
import math
from pathlib import Path

import pandas as pd

from geo_utils import extract_town, geocode_town

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw" / "ocds"
PROCESSED_DIR = DATA_DIR / "processed"

EUR_BGN = 1.95583
MIN_DAYS, MAX_DAYS = 5, 1800  # drop implausible / data-entry outliers

# CPV division 45 sub-groups -> readable category (for display + filtering)
CPV_LABELS = {
    "4520": "site_prep_demolition",
    "4521": "building_construction",
    "4522": "civil_engineering",
    "4523": "roads_highways",
    "4524": "water_marine_works",
    "4525": "other_civil_works",
    "4526": "roof_structural",
    "4531": "electrical_installation",
    "4532": "insulation_works",
    "4533": "plumbing_heating",
    "4534": "fencing_railing",
    "4500": "general_construction",
}

# EU funding signals in the contract title (best-effort, OCDS lacks a flag)
EU_KEYWORDS = ("оперативна програма", "европейск", "ефрр", "еф рр", "опрр",
               "осес", "пррд", "приоритет", "съфинансир", "ес ")


def season(m: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"}.get(m, "unknown")


def method_group(d: str) -> str:
    d = (d or "").lower()
    if "открит" in d or "open" in d:
        return "open"
    if "събиране" in d or "collect" in d or "selective" in d:
        return "collect_offers"
    if "пряко" in d or "direct" in d or "negoti" in d:
        return "direct"
    if "ограничен" in d or "limited" in d or "restrict" in d:
        return "restricted"
    return "other"


def buyer_type(name: str) -> str:
    n = (name or "").lower()
    if any(w in n for w in ["община", "кметство", "район"]):
        return "municipality"
    if any(w in n for w in ["министерств", "агенци", "област", "държав"]):
        return "state"
    if any(w in n for w in ["еоод", "еад", " ад", "оод", "вик", "топлофикац"]):
        return "utility"
    if any(w in n for w in ["училищ", "детска", "болниц", "университет", "читалищ"]):
        return "institution"
    return "other"


def cpv_label(cpv: str) -> str:
    return CPV_LABELS.get(cpv[:4], "other_construction")


def parse_date(s):
    if not s:
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def iter_releases():
    files = sorted(glob.glob(str(RAW_DIR / "*.jsonl.gz")))
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def build_rows():
    rows = []
    for rec in iter_releases():
        t = rec.get("tender", {}) or {}
        cp = t.get("contractPeriod") or {}
        dur = cp.get("durationInDays")
        if not isinstance(dur, (int, float)) or not (MIN_DAYS <= dur <= MAX_DAYS):
            continue

        items = t.get("items") or []
        cpv = str((items[0].get("classification") or {}).get("id", "")) if items else ""
        if not cpv.startswith("45"):  # construction / infrastructure only
            continue

        val = t.get("value") or {}
        amt = val.get("amount") or 0
        if (val.get("currency") or "BGN").upper() == "EUR":
            amt *= EUR_BGN
        if not amt or amt <= 0:
            continue

        tp = t.get("tenderPeriod") or {}
        d = parse_date(tp.get("endDate")) or parse_date(rec.get("date"))
        month = int(d.month) if d is not None else 0

        buyer = (rec.get("buyer") or {}).get("name", "")
        postal = ""
        for p in (rec.get("parties") or []):
            a = p.get("address") or {}
            if "buyer" in (p.get("roles") or []) and a.get("postalCode"):
                postal = str(a["postalCode"])
                break
            if a.get("postalCode") and not postal:
                postal = str(a["postalCode"])

        town = extract_town(buyer)
        lat, lng = geocode_town(town)
        title = (t.get("title") or t.get("description") or "").strip()
        eu = 1 if any(k in title.lower() for k in EU_KEYWORDS) else 0

        rows.append(dict(
            id=rec.get("ocid", ""),
            contract_number=str(t.get("id", "")),
            subject=title[:300],
            repair_type=cpv_label(cpv),
            cpv4=cpv[:4],
            object_type=t.get("mainProcurementCategory", "works"),
            method=method_group(t.get("procurementMethodDetails")),
            season=season(month),
            month=month,
            authority=buyer,
            authority_type=buyer_type(buyer),
            buyer_type=buyer_type(buyer),
            town=town,
            lat=lat,
            lng=lng,
            value_bgn=round(float(amt), 2),
            value_log=math.log1p(amt),
            num_offers=len(rec.get("bids") or []),
            postal_region=(postal[:2] if postal else "??"),
            eu_financed=eu,
            contracted_days=float(dur),
            duration_source="real_ocds",
        ))
    return rows


def main():
    print("=" * 60)
    print("  OCDS Data Cleaning & Feature Engineering")
    print("=" * 60)

    rows = build_rows()
    if not rows:
        print("ERROR: no rows produced. Run fetch_data.py first.")
        return

    df = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "contracts_clean.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n  Construction contracts with REAL duration: {len(df)}")
    print(f"  contracted_days mean={df.contracted_days.mean():.0f} "
          f"median={df.contracted_days.median():.0f} std={df.contracted_days.std():.0f}")
    print(f"  repair_type: {df.repair_type.value_counts().head(6).to_dict()}")
    print(f"  Saved -> {out}")

    meta = {
        "repair_types": sorted(df["repair_type"].unique().tolist()),
        "object_types": sorted(df["object_type"].unique().tolist()),
        "methods": sorted(df["method"].unique().tolist()),
        "seasons": ["winter", "spring", "summer", "autumn"],
        "authority_types": sorted(df["authority_type"].unique().tolist()),
        "towns": sorted([t for t in df["town"].unique().tolist() if t and t != "неизвестен"]),
        "num_samples": len(df),
        "mean_days": float(df["contracted_days"].mean()),
        "target": "contracted_days",
        "target_note": "Real contracted execution period (durationInDays) from OCDS; "
                       "duration agreed at signing, not actual completion time.",
    }
    fm = PROCESSED_DIR / "feature_metadata.json"
    with open(fm, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Feature metadata saved -> {fm}")


if __name__ == "__main__":
    main()
