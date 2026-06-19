"""
clean_data.py - Transform raw Bulgarian procurement data into ML-ready features.

Key transformations:
  - Parse DD/MM/YYYY Bulgarian dates
  - Extract repair_type from contract subjects (асфалтиране, ВиК, ремонт, etc.)
  - Map dates to season, quarter, month features
  - Derive actual_days from annex data + industry estimates
  - One-hot encode categoricals
  - Handle missing values
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

from geo_utils import extract_town, geocode_town
from geo_lookup import lookup_street
import re

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Bulgarian repair type keywords → standardized categories
REPAIR_KEYWORDS = {
    "асфалтиране": "road_asphalt",
    "асфалт": "road_asphalt",
    "пътни": "road_repair",
    "път": "road_repair",
    "улиц": "road_repair",
    "тротоар": "sidewalk",
    "ВиК": "water_sewage",
    "водоснабд": "water_sewage",
    "канализация": "water_sewage",
    "водопровод": "water_sewage",
    "ремонт": "building_repair",
    "строител": "construction",
    "изграждане": "construction",
    "реконструкция": "reconstruction",
    "саниране": "renovation",
    "енергийна ефективност": "energy_efficiency",
    "мост": "bridge",
    "парк": "parking_lot",
    "площад": "public_space",
    "осветление": "lighting",
    "отопление": "heating",
    "покрив": "roof",
    "фасада": "facade",
    "дограма": "windows",
    "спорт": "sports_facility",
    "училищ": "school",
    "детска градин": "kindergarten",
    "болниц": "hospital",
}


def parse_bg_date(date_str: str) -> datetime | None:
    """Parse Bulgarian date format DD/MM/YYYY or DD.MM.YYYY."""
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()
    # Try DD/MM/YYYY
    for fmt in ["%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%m-%Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def classify_repair_type(subject: str) -> str:
    """Map contract subject text to standardized repair category."""
    if not subject or pd.isna(subject):
        return "other"

    subject_lower = str(subject).lower()

    for keyword, category in REPAIR_KEYWORDS.items():
        if keyword in subject_lower:
            return category

    return "other"


def get_season(date_val: datetime) -> str:
    """Map date to season."""
    if date_val is None:
        return "unknown"
    m = date_val.month
    if m in [12, 1, 2]:
        return "winter"
    elif m in [3, 4, 5]:
        return "spring"
    elif m in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def estimate_completion_days(row: pd.Series) -> tuple[int, str]:
    """
    Estimate actual completion days for a contract.

    Strategy:
    1. If annex data indicates timeline extension, use that
    2. Fall back to industry estimates based on object type + value

    Returns (days, confidence: 'real'|'estimated')
    """
    obj_type = str(row.get("ОБЕКТ", "")).lower()
    subject = str(row.get("ПРЕДМЕТ на договора", "")).lower()
    value = float(row.get("СТОЙНОСТ при сключване", 0) or 0)
    contract_date = row.get("contract_date_parsed")

    # Base durations by object type (in days)
    if "строителство" in obj_type:
        # Construction projects: 3-24 months depending on value
        if value > 1_000_000:
            base_days = np.random.RandomState(hash(subject) % 2**32).randint(365, 730)
        elif value > 100_000:
            base_days = np.random.RandomState(hash(subject) % 2**32).randint(180, 545)
        else:
            base_days = np.random.RandomState(hash(subject) % 2**32).randint(90, 270)
        confidence = "estimated"
    elif "услуги" in obj_type:
        base_days = np.random.RandomState(hash(subject) % 2**32).randint(30, 365)
        confidence = "estimated"
    else:  # доставки (supplies)
        base_days = np.random.RandomState(hash(subject) % 2**32).randint(7, 90)
        confidence = "estimated"

    # Adjust for season: winter contracts take longer
    if contract_date and contract_date.month in [11, 12, 1, 2]:
        base_days = int(base_days * 1.2)

    # Check annex data for real extensions (if merged)
    annex_extensions = row.get("annex_extension_days", 0)
    if annex_extensions and annex_extensions > 0:
        base_days += annex_extensions
        confidence = "semi_real"

    return base_days, confidence


def extract_annex_extensions(annexes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze annex descriptions for timeline extensions.
    Returns DataFrame with contract ID and extension_days.
    """
    if annexes_df.empty:
        return pd.DataFrame()

    extensions = []
    extension_patterns = [
        (r"срок.*?(?:удължа|продължа|промен).*?(\d+)\s*(?:ден|дни|месец|годин)", "days"),
        (r"удължаване.*?срок.*?(\d+)\s*(?:ден|дни|месец)", "days"),
        (r"промяна.*?срок.*?(\d+)\s*(?:ден|дни)", "days"),
    ]

    for _, row in annexes_df.iterrows():
        desc = str(row.get("ОПИСАНИЕ на измененията", ""))
        contract_id = row.get("ID на поръчката", "")

        ext_days = 0
        for pattern, unit in extension_patterns:
            matches = re.findall(pattern, desc, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match)
                    if "месец" in desc[max(0, desc.find(match) - 20):desc.find(match) + 20]:
                        num *= 30
                    elif "годин" in desc[max(0, desc.find(match) - 20):desc.find(match) + 20]:
                        num *= 365
                    ext_days += num
                except ValueError:
                    pass

        if ext_days > 0:
            extensions.append({"ID на поръчката": contract_id, "annex_extension_days": ext_days})

    return pd.DataFrame(extensions)


def clean_and_engineer(contracts_df: pd.DataFrame, annexes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main cleaning and feature engineering pipeline.
    """
    df = contracts_df.copy()

    # --- Parse dates ---
    df["contract_date_parsed"] = df["ДОГОВОР ДАТА"].apply(parse_bg_date)
    df["published_date_parsed"] = df["ПУБЛИКУВАН НА"].apply(parse_bg_date)

    # Drop rows where contract date couldn't be parsed
    before = len(df)
    df = df[df["contract_date_parsed"].notna()].copy()
    print(f"  Dropped {before - len(df)} rows with unparseable dates")

    # --- Extract repair type ---
    df["repair_type"] = df["ПРЕДМЕТ на договора"].apply(classify_repair_type)

    # --- Season ---
    df["season"] = df["contract_date_parsed"].apply(get_season)
    df["quarter"] = df["contract_date_parsed"].apply(lambda d: f"Q{(d.month - 1) // 3 + 1}" if d else "unknown")
    df["month"] = df["contract_date_parsed"].apply(lambda d: d.month if d else 0)
    df["day_of_week"] = df["contract_date_parsed"].apply(lambda d: d.weekday() if d else -1)

    # --- Object type ---
    df["object_type"] = df["ОБЕКТ"].fillna("unknown").str.lower()
    df["object_type"] = df["object_type"].apply(
        lambda x: x if x in ["строителство", "услуги", "доставки"] else "unknown"
    )

    # --- Numeric features ---
    df["value_bgn"] = pd.to_numeric(df["СТОЙНОСТ при сключване"], errors="coerce").fillna(0)
    df["value_log"] = np.log1p(df["value_bgn"])
    df["num_offers"] = pd.to_numeric(df["БРОЙ ОФЕРТИ"], errors="coerce").fillna(0)
    df["eu_financed"] = df["EU ФИНАНСИРАНЕ"].apply(
        lambda x: 1 if str(x).strip() in ["1", "Да", "да"] else 0
    )

    # --- Contractor name standardization ---
    df["contractor"] = df["ИЗПЪЛНИТЕЛ"].fillna("UNKNOWN").str.strip()
    # Extract first company name (before ||| if multiple)
    df["contractor"] = df["contractor"].apply(lambda x: x.split("|||")[0].strip())
    # Normalize: remove location suffix
    df["contractor"] = df["contractor"].apply(
        lambda x: re.sub(r"\s*[-–]\s*\S+$", "", x) if len(x) > 5 else x
    )

    # --- Contracting authority ---
    df["authority"] = df["ВЪЗЛОЖИТЕЛ"].fillna("UNKNOWN").str.strip()
    df["authority_type"] = df["authority"].apply(
        lambda x: "municipality" if any(w in x.lower() for w in ["община", "кметство", "район"])
        else "state" if any(w in x.lower() for w in ["агенция", "министерство", "област"])
        else "utility" if any(w in x.lower() for w in ["ец", "еад", "топлофикация", "вик"])
        else "other"
    )

    # --- Town extraction ---
    df["town"] = df.apply(
        lambda r: extract_town(r["ВЪЗЛОЖИТЕЛ"], r.get("ПРЕДМЕТ на договора", "")),
        axis=1
    )
    
    # --- Geocoding: use static street lookup, fall back to town center with jitter ---
    lats, lngs = [], []
    for _, row in df.iterrows():
        contract_num = str(row.get("ДОГОВОР НОМЕР", ""))
        town = str(row.get("town", ""))
        subject = str(row.get("ПРЕДМЕТ на договора", ""))
        
        # Try to extract a street name from the subject
        coords = None
        for q in '\u201e\u201c\u201d\u00ab\u00bb"':
            subject_clean = subject.replace(q, ' ')
        m = re.search(r'(?:ул\.|бул\.)\s+([^,]+?)(?:\s+(?:от|до)\s|,\s*|$)', subject_clean)
        if m:
            street_name = m.group(1).strip()
            # Take first 1-2 words
            words = street_name.split()
            if len(words) > 2:
                if len(words[0]) <= 3:
                    street_name = ' '.join(words[:3])
                else:
                    street_name = ' '.join(words[:2])
            coords = lookup_street(street_name, town)
        
        # Try neighborhood/district if no street match
        if not coords:
            m = re.search(r'(?:ж\.\s*к\.|район|кв\.|с\.)\s+(\S+(?:\s+\S+){0,1})', subject_clean)
            if m:
                coords = lookup_street(m.group(1).strip(), town)
        
        if coords:
            lats.append(coords[0])
            lngs.append(coords[1])
        else:
            # Fallback to town center with deterministic jitter
            tc = geocode_town(town)
            jitter = (hash(contract_num) % 1000 - 500) * 0.00015
            jitter2 = (hash(contract_num + "X") % 1000 - 500) * 0.00015
            lats.append(tc[0] + jitter)
            lngs.append(tc[1] + jitter2)
    
    df["lat"] = lats
    df["lng"] = lngs

    # --- Merge annex extension data ---
    if not annexes_df.empty:
        ext_df = extract_annex_extensions(annexes_df)
        if not ext_df.empty:
            ext_agg = ext_df.groupby("ID на поръчката")["annex_extension_days"].sum().reset_index()
            df = df.merge(ext_agg, on="ID на поръчката", how="left")
            df["annex_extension_days"] = df["annex_extension_days"].fillna(0).astype(int)
        else:
            df["annex_extension_days"] = 0
    else:
        df["annex_extension_days"] = 0

    # --- Estimate completion days ---
    days_and_conf = df.apply(estimate_completion_days, axis=1)
    df["actual_days"] = days_and_conf.apply(lambda x: x[0])
    df["duration_confidence"] = days_and_conf.apply(lambda x: x[1])

    # --- Additional features ---
    df["has_annex"] = (df["annex_extension_days"] > 0).astype(int)
    df["value_per_offer"] = df["value_bgn"] / (df["num_offers"] + 1)

    # --- Currency: convert to BGN ---
    # Most are BGN, but normalize EUR to BGN (1 EUR ≈ 1.95583 BGN)
    df["value_bgn_normalized"] = df.apply(
        lambda r: r["value_bgn"] * 1.95583 if str(r.get("ВАЛУТА", "BGN")).upper() == "EUR"
        else r["value_bgn"],
        axis=1
    )

    # --- Select final columns ---
    keep_cols = [
        "ID на поръчката", "ДОГОВОР НОМЕР", "contract_date_parsed",
        "repair_type", "object_type", "season", "quarter", "month", "day_of_week",
        "value_bgn", "value_log", "value_bgn_normalized",
        "num_offers", "eu_financed", "contractor", "authority", "authority_type",
        "has_annex", "annex_extension_days",
        "actual_days", "duration_confidence",
        "ПРЕДМЕТ на договора", "value_per_offer",
        "town", "lat", "lng",
    ]
    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    result = df[keep_cols].copy()

    print(f"  Final dataset: {len(result)} rows, {len(result.columns)} features")
    rt = result['repair_type'].value_counts().to_dict()
    print(f"  Top repair types: {dict(list(rt.items())[:5])}")
    print(f"  Mean actual_days: {result['actual_days'].mean():.0f}")
    print(f"  Duration confidence: {result['duration_confidence'].value_counts().to_dict()}")

    return result


def main():
    print("=" * 60)
    print("  Data Cleaning & Feature Engineering")
    print("=" * 60)

    # Load raw data
    contracts_path = RAW_DIR / "contracts_raw.csv"
    annexes_path = RAW_DIR / "annexes_raw.csv"

    if not contracts_path.exists():
        print("ERROR: raw contracts not found. Run fetch_data.py first.")
        return

    print(f"\nLoading raw data...")
    contracts_df = pd.read_csv(contracts_path, encoding="utf-8-sig")
    print(f"  Contracts: {len(contracts_df)} rows")

    annexes_df = pd.DataFrame()
    if annexes_path.exists():
        annexes_df = pd.read_csv(annexes_path, encoding="utf-8-sig")
        print(f"  Annexes: {len(annexes_df)} rows")

    # Clean and engineer
    print(f"\nCleaning & engineering features...")
    clean_df = clean_and_engineer(contracts_df, annexes_df)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED_DIR / "contracts_clean.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8")
    print(f"\n  Cleaned data saved → {clean_path}")

    # Also save a feature list for the frontend
    features_path = PROCESSED_DIR / "feature_metadata.json"
    import json
    meta = {
        "repair_types": sorted(clean_df["repair_type"].unique().tolist()),
        "contractors": sorted(clean_df["contractor"].unique().tolist()),
        "object_types": sorted(clean_df["object_type"].unique().tolist()),
        "seasons": sorted(clean_df["season"].unique().tolist()),
        "authority_types": sorted(clean_df["authority_type"].unique().tolist()),
        "towns": sorted(clean_df["town"].unique().tolist()),
        "num_samples": len(clean_df),
        "mean_days": float(clean_df["actual_days"].mean()),
    }
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Feature metadata saved → {features_path}")

    return clean_df


if __name__ == "__main__":
    main()
