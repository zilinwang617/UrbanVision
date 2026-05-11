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
│   └── Tax.py                # WPRDC tax-status queries
├── notebook/
│   ├── gsv-data.ipynb        # Bulk image download workflow
│   ├── Data_cleaning.ipynb   # Raw data preprocessing
│   ├── query_properties.py   # Vision model evaluation
│   ├── evaluation.ipynb      # Results analysis
│   └── visualization.ipynb   # Spatial heatmaps
├── data/
│   ├── cleaned_data.csv      # Processed property records
│   └── raw.csv               # Raw parcel data
├── result/                   # Model prediction outputs (JSON)
├── .env.example
└── requirements.txt
```

## Usage

### 1. Download Street View images

Run `notebook/gsv-data.ipynb` or call the pipeline directly:

```python
from src.gsv_pipeline import fetch_entries
fetch_entries("data/cleaned_data.csv", out_dir="data/gsv_out", start=0, end=1000)
```

Key parameters in `fetch_entries`:
- `search_radius` — panorama search radius in meters (default 50)
- `max_distance` — hard cutoff for selected pano distance (default 20 m)
- `year_tolerance` — how many years from property creation date to allow (default 2)

### 2. Evaluate properties

Run `notebook/query_properties.py` to send images to the vision model and get JSON assessments (occupancy status, damage indicators, confidence score).

### 3. Analyze & visualize

Open `notebook/evaluation.ipynb` for accuracy metrics across prompt variants, and `notebook/visualization.ipynb` for spatial heatmaps of predicted abandonment.

## Results

Current scale: **1,000 properties** evaluated. Outputs in `result/`:
- `result_1-1000.json` — baseline prompt results
- `new-prompt-result.json` / `refined_propmpt_results.json` — prompt iteration results
