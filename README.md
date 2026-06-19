# Infrastructure Repair Predictor 🇧🇬

Interactive map-based ML dashboard that predicts municipal infrastructure repair timelines in Bulgaria.

**Live:** [infrastructure.stormlabs.cloud](http://infrastructure.stormlabs.cloud)

## What it does

- **186 real Bulgarian public procurement contracts** from [data.egov.bg](https://data.egov.bg)
- **12 towns** mapped with street-level geographic accuracy
- **XGBoost model** predicts repair duration (MAE ±79 days)
- **Color-coded map** — green/yellow/red pins by delay risk
- **Town filter** — select a municipality to see its active repairs

## Tech Stack

| Layer | Tech |
|-------|------|
| Model | XGBoost, scikit-learn, pandas |
| Backend | FastAPI (Python) |
| Frontend | Leaflet.js, vanilla JS |
| Data | Bulgarian Open Data Portal |
| Serving | Nginx Proxy Manager, systemd |

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Fetch & train
cd backend
python fetch_data.py
python clean_data.py
python train.py

# Serve (frontend + API on port 3000)
python -m uvicorn main:app --host 0.0.0.0 --port 3000
```

## Project Structure

```
backend/
├── fetch_data.py      # Download from data.egov.bg
├── clean_data.py      # Feature engineering + geocoding
├── train.py           # XGBoost training pipeline
├── main.py            # FastAPI server (serves frontend + API)
├── geo_lookup.py      # Static geocoding for BG streets
└── data/
    ├── raw/           # Raw CSVs (186 contracts, 625 annexes)
    └── processed/     # Cleaned data + feature metadata

frontend/
└── src/               # React/TS source (API client + components)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/repairs?town=варна` | GeoJSON repairs with predictions |
| GET | `/api/meta` | Feature categories for UI |
| POST | `/api/predict` | Single repair prediction |
| GET | `/` | Map dashboard |

## Data Sources

Bulgarian Public Procurement Agency (AOP) via the [National Open Data Portal](https://data.egov.bg) — 2024-2025 contracts with annexes, geocoded to street/district level.
