# Infrastructure Contract Duration Predictor 🇧🇬

Predicts the **contracted execution period** of Bulgarian construction/infrastructure
procurement contracts from their published attributes, using a real, open dataset.

**Live:** [infrastructure.stormlabs.cloud](http://infrastructure.stormlabs.cloud)

## What it predicts (and what it doesn't)

The model predicts `contractPeriod.durationInDays` — the execution period the
contracting authority **agreed at signing**. This is a real field published in the
open data, not a synthesised value.

> ⚠️ It does **not** predict actual on-site completion time or delay. Bulgarian open
> procurement data does not publish actual completion dates, so "how long the repair
> really took" is not modellable from this source. Earlier versions of this project
> fabricated that target with a random number generator; that has been removed.

## Data

- **Source:** Open Contracting Data Registry — Bulgaria, published by DIGIWHIST /
  [opentender.eu](https://opentender.eu/bg) and mirrored at
  [data.open-contracting.org/en/publication/44](https://data.open-contracting.org/en/publication/44)
- **Format:** OCDS (Open Contracting Data Standard) JSON-Lines
- **Scope:** CPV division 45 (construction works), years 2020–2023
- **Rows:** ~12,600 contracts with a real `durationInDays` target
- **License:** **CC BY-NC-SA 4.0** — attribution, non-commercial, share-alike (see `DATA_LICENSE.md`)

## Model

XGBoost regression on real features:

| Type | Features |
|------|----------|
| Categorical | CPV4 group, procurement method, buyer type, season, postal region |
| Numeric | log(contract value), number of bids, contract month |

**Honest metrics** (20% hold-out test, 5-fold CV):

| Metric | Model | Mean-predictor baseline |
|--------|-------|-------------------------|
| MAE | **91.5 days** | 131.6 days |
| RMSE | 159.3 | 227.8 |
| R² (test) | **0.51** | ~0.00 |
| R² (train) | 0.59 | — |

The model beats the naive "always predict the mean" baseline by **~30% MAE**, with a
small train/test R² gap (0.59 → 0.51). 5-fold CV MAE ≈ 95 ± 4 days. These numbers are
persisted to `backend/model_meta.json` on every training run.

## Tech Stack

| Layer | Tech |
|-------|------|
| Model | XGBoost, scikit-learn, pandas |
| Backend | FastAPI (Python) |
| Frontend | React + TypeScript (Vite) |
| Data | Open Contracting / OCDS (DIGIWHIST Bulgaria) |
| Serving | systemd + reverse proxy |

## Quick Start

```bash
pip install -r backend/requirements.txt
cd backend
python fetch_data.py    # downloads OCDS yearly files (~84 MB) into data/raw/ocds/
python clean_data.py    # -> data/processed/contracts_clean.csv
python train.py         # -> model.joblib + model_meta.json (prints honest metrics)
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Model target + real test metrics |
| GET | `/api/meta` | Feature categories for the UI |
| GET | `/api/repairs?town=ловеч` | Contracts with predicted contracted duration |
| POST | `/api/predict` | Single prediction |
| GET | `/` | Dashboard (if `frontend/dist` is built) |

CORS is restricted to the configured frontend origin(s); override with the
`ALLOWED_ORIGINS` environment variable (comma-separated).

## Attribution

Contains information from the Open Contracting Data Registry / DIGIWHIST
(opentender.eu), licensed under CC BY-NC-SA 4.0. This project is non-commercial.
