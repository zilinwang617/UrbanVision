# UrbanVision

Identify potentially abandoned residential properties in Pittsburgh using Google Street View imagery and a vision language model.

## Overview

UrbanVision fetches street-level photos for residential parcels, passes them to a Qwen vision model for occupancy and condition assessment, and aggregates the results into spatial heatmaps. The pipeline is designed for the Pittsburgh metro area using WPRDC parcel and tax-status data.

## Tech Stack

- **Python** — pandas, geopandas, matplotlib, scipy, numpy
- **Google Street View Static API** — image acquisition
- **Google Geocoding API** — address → coordinates
- **Qwen2.5-VL-32B** vision model (OpenAI-compatible endpoint)
- **WPRDC Property API** — tax status enrichment

## Setup

```bash
git clone <repo-url>
cd UrbanVision
pip install -r requirements.txt
cp .env.example .env          # then fill in your key
```

`.env` variables:

| Variable | Description |
|---|---|
| `GSV_API_KEY` | Google Street View / Geocoding API key |

The vision model is expected at `http://localhost:8000/v1` (vLLM or compatible server).

## Project Structure

```
UrbanVision/
├── src/
│   ├── gsv_pipeline.py       # Street View image acquisition pipeline
│   └── tax.py                # WPRDC tax-status queries
├── notebook/
│   ├── gsv_data.ipynb        # Bulk image download workflow
│   ├── data_cleaning.ipynb   # Raw data preprocessing
│   ├── query_properties.py   # Vision model evaluation
│   ├── evaluation.ipynb      # Results analysis
│   └── visualization.ipynb   # Spatial heatmaps
├── data/
│   ├── cleaned_data.csv      # Processed property records
│   └── raw.csv               # Raw parcel data
├── result/                   # Pipeline outputs (gitignored)
│   └── <run_name>.json       # Per-property assessment: status, damage indicators, confidence
├── .env.example
└── requirements.txt
```

Each entry in a result JSON contains:
- `parcel_id` — property identifier
- `status` — `occupied` / `abandoned` / `uncertain`
- `indicators` — list of observed signals (boarded windows, overgrowth, graffiti, etc.)
- `confidence` — model confidence score

## Usage

### 1. Download Street View images

Run `notebook/gsv_data.ipynb` or call the pipeline directly:

```python
from src.gsv_pipeline import fetch_entries
fetch_entries("data/cleaned_data.csv", out_dir="data/gsv_out", start=0, end=1000)
```

Key parameters in `fetch_entries`:
- `search_radius` — panorama search radius in meters (default 50)
- `max_distance` — hard cutoff for selected pano distance (default 20 m)
- `year_tolerance` — how many years from property creation date to allow (default 2)

Images and metadata are saved under `data/gsv_out/<entry_id>/`.

### 2. Evaluate properties

Run `notebook/query_properties.py` to send images to the vision model. Results are written to `result/<run_name>.json`.

### 3. Analyze & visualize

Open `notebook/evaluation.ipynb` for accuracy metrics across prompt variants, and `notebook/visualization.ipynb` for spatial heatmaps of predicted abandonment.

## License

Copyright (c) 2025 Zilin Wang, Samuel Houpt, and Xiaoting Wang. All Rights Reserved.
