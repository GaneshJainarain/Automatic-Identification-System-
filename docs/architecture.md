# Architecture

```mermaid
flowchart TD
    subgraph SRC["1. Data Sources"]
        AIS["NOAA MarineCadastre AIS<br/>daily CSV zips (2023)"]
        REF["Reference data<br/>ports, coastline, shipping lanes"]
        REG["Vessel static info<br/>type, length, IMO"]
        LIVE["AISStream.io<br/>live AIS WebSocket"]
    end

    subgraph ETL["2. Ingest & Clean (ETL)"]
        DL["Download + region filter<br/>(bbox, e.g. Florida Straits)"]
        VAL["Validate & clean<br/>bad MMSI, bad coords, sentinels,<br/>duplicates, GPS glitches"]
        SEG["Track segmentation<br/>split on time gaps, record gap length"]
        RPT["Cleaning report<br/>rows dropped per rule"]
    end

    subgraph STORE["3. Storage"]
        PQ[("Parquet<br/>raw / clean / features")]
        PG[("PostgreSQL + PostGIS<br/>positions, vessels, tracks, alerts")]
    end

    subgraph FEAT["4. Feature Engineering"]
        KIN["Kinematic features<br/>speed, accel, turn rate, heading change"]
        GEO["Geospatial features<br/>distance to shore / port / lane"]
        CTX["Context features<br/>vessel type, time of day, nearby vessels"]
    end

    subgraph LABEL["5. Training / Test Data"]
        INJ["Synthetic anomaly injection<br/>loiter, going dark, rendezvous,<br/>spoofed jumps, route deviation"]
        SPLIT["Train / test split<br/>by vessel and by time"]
    end

    subgraph MODELS["6. Detectors"]
        RULE["Rule & statistical baselines<br/>thresholds, z-scores, change points"]
        ML["Classical ML<br/>Isolation Forest, HDBSCAN routes,<br/>gradient boosting"]
        DLM["Deep learning (PyTorch)<br/>LSTM / Transformer autoencoder"]
    end

    subgraph EVAL["7. Evaluation"]
        MET["Scoring<br/>precision / recall, PR-AUC,<br/>detection delay, false alarms per 1k tracks"]
        CMP["Model comparison<br/>+ statistical significance"]
    end

    subgraph SIM["8. Operational Simulation"]
        NORM["Common schema adapter<br/>live + historical → same columns"]
        REPLAY["Streaming scorer<br/>live feed, or NOAA replay fallback"]
        TUNE["Alert threshold tradeoffs<br/>alert volume vs detection"]
    end

    subgraph OUT["9. Outputs"]
        DASH["Dashboard<br/>map of tracks + alerts"]
        REPORT["Analytic report<br/>methods, results, recommendations"]
    end

    AIS --> DL --> VAL --> SEG
    VAL --> RPT
    SEG --> PQ
    SEG --> PG
    REG --> PG
    REF --> GEO

    PQ --> KIN & GEO & CTX
    PG --> CTX
    KIN & GEO & CTX --> INJ --> SPLIT

    SPLIT --> RULE & ML & DLM
    RULE & ML & DLM --> MET --> CMP

    LIVE --> NORM --> REPLAY
    PQ --> NORM
    CMP -->|trained models| REPLAY --> TUNE
    REPLAY --> PG
    PG --> DASH
    CMP --> REPORT
    TUNE --> REPORT
```

## Code layout (planned)

| Stage | Module |
|---|---|
| Ingest | `src/kad/ingest.py` |
| Clean + segment | `src/kad/clean.py` |
| Storage | `src/kad/db.py`, `sql/schema.sql` |
| Features | `src/kad/features/` |
| Synthetic anomalies | `src/kad/synth.py` |
| Detectors | `src/kad/models/` (`rules.py`, `ml.py`, `deep.py`) |
| Evaluation | `src/kad/evaluate.py` |
| Live feed | `src/kad/live.py` |
| Simulation | `src/kad/simulate.py` |
| Dashboard | `app/` (Streamlit) |
