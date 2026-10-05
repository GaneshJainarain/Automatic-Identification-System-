"""Download NOAA MarineCadastre daily AIS files and keep only one region."""

import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from kad import schema
from kad.config import DATA_DIR, REGIONS, BBox

log = logging.getLogger(__name__)

NOAA_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{day:%Y}/AIS_{day:%Y_%m_%d}.zip"


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)


def download_day(day: date, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"AIS_{day:%Y_%m_%d}.zip"
    if path.exists():
        return path
    url = NOAA_URL.format(day=day)
    log.info("downloading %s", url)
    tmp = path.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            fh.writelines(resp.iter_content(chunk_size=1 << 20))
    tmp.rename(path)
    return path


def read_region(zip_path: Path, bbox: BBox, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the CSV inside a daily zip, keeping only rows inside bbox."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            parts = [
                chunk[bbox.contains(chunk["LAT"], chunk["LON"])]
                for chunk in pd.read_csv(fh, dtype=schema.NOAA_DTYPES, chunksize=chunksize)
            ]
    return schema.from_noaa(pd.concat(parts, ignore_index=True))


def raw_dir(region: str) -> Path:
    return DATA_DIR / "raw" / region


def ingest_days(start: date, end: date, region: str, keep_zips: bool = True) -> list[Path]:
    bbox = REGIONS[region]
    out_dir = raw_dir(region)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for day in daterange(start, end):
        out = out_dir / f"{day:%Y-%m-%d}.parquet"
        if out.exists():
            log.info("skip %s (exists)", out)
            written.append(out)
            continue
        zip_path = download_day(day, DATA_DIR / "zips")
        df = read_region(zip_path, bbox)
        df.to_parquet(out, index=False)
        log.info("%s: %d rows, %d vessels", day, len(df), df["mmsi"].nunique())
        if not keep_zips:
            zip_path.unlink()
        written.append(out)
    return written


def load_raw(region: str, start: date, end: date) -> pd.DataFrame:
    paths = [raw_dir(region) / f"{d:%Y-%m-%d}.parquet" for d in daterange(start, end)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"run `kad ingest` first; missing {missing[0]} (+{len(missing) - 1})"
        )
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
