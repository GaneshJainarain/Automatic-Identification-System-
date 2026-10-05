# Automatic Identification System — Kinematic Anomaly Detection

Detecting suspicious vessel behavior from AIS (Automatic Identification System) ship-tracking
data: loitering, going dark (AIS gaps), at-sea rendezvous, spoofed position jumps, and route
deviation. Models are trained and scored on historical NOAA data and run against a live AIS feed.

See [docs/architecture.md](docs/architecture.md) for the full design.

## Data

- **Historical:** [NOAA MarineCadastre](https://hub.marinecadastre.gov/pages/vesseltraffic)
  daily AIS files (2023), ~320 MB zipped per day for all US waters, filtered to a region.
- **Live:** [AISStream.io](https://aisstream.io) WebSocket feed (planned, phase 6).

Default region: **Florida Straits** (lat 23.0–27.5, lon −83.5 to −79.0).

## Quickstart

```bash
python -m venv env && env/bin/pip install -e ".[dev]"
docker compose up -d                    # PostgreSQL + PostGIS

kad ingest --start 2023-01-01 --end 2023-01-07   # download + filter to region -> data/raw/
kad clean  --start 2023-01-01 --end 2023-01-07   # validate + segment -> data/clean/
kad load                                          # -> PostGIS
pytest
```

## Cleaning rules

Each rule's drop count is written to `data/clean/<region>/cleaning_report.json`.

| Rule | Action |
|---|---|
| Invalid MMSI | Drop anything not a 9-digit ship ID with a valid country prefix (201–775) |
| Invalid coordinates | Drop out-of-range and (0, 0) positions |
| "Not available" values | SOG 102.3, COG 360, heading 511 → null |
| Duplicate timestamps | Keep first report per vessel per timestamp |
| GPS glitches | Drop single fixes with an implausible jump out *and* back (> 50 kn, > 1 km) |
| Track segmentation | New track after a gap > 30 min; gap length kept as `gap_before_s` |
| Short tracks | Drop tracks with < 10 fixes |

Sustained position jumps (no jump back) are **kept** — that pattern is a spoofing signal for
the detectors, not noise.

### Example: Florida Straits, 2023-01-01

| | |
|---|---|
| Raw positions | 717,773 (2,418 vessels) |
| Invalid MMSI | 2,343 |
| Duplicates | 30 |
| GPS glitches | 97 |
| Short-track fixes | 8,129 |
| **Clean positions** | **707,174 (2,133 vessels, 2,917 tracks)** |

## Roadmap

1. ✅ Ingest, clean, store (Parquet + PostGIS)
2. Kinematic / geospatial features + rule-based baselines
3. Synthetic anomaly injection + evaluation framework
4. Classical ML (Isolation Forest, HDBSCAN, gradient boosting)
5. Deep learning (PyTorch sequence autoencoder)
6. Live AISStream scorer + dashboard
7. Analytic report
