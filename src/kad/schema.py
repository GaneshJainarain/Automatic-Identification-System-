"""Canonical column schema shared by historical (NOAA) and live (AISStream) sources."""

import pandas as pd

NOAA_COLUMNS = {
    "MMSI": "mmsi",
    "BaseDateTime": "ts",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "IMO": "imo",
    "CallSign": "call_sign",
    "VesselType": "vessel_type",
    "Status": "nav_status",
    "Length": "length",
    "Width": "width",
    "Draft": "draft",
    "Cargo": "cargo",
    "TransceiverClass": "transceiver_class",
}

NOAA_DTYPES = {
    "MMSI": "int64",
    "BaseDateTime": "string",
    "LAT": "float64",
    "LON": "float64",
    "SOG": "float32",
    "COG": "float32",
    "Heading": "float32",
    "VesselName": "string",
    "IMO": "string",
    "CallSign": "string",
    "VesselType": "Int16",
    "Status": "Int16",
    "Length": "float32",
    "Width": "float32",
    "Draft": "float32",
    "Cargo": "Int16",
    "TransceiverClass": "string",
}

POSITION_COLUMNS = ["mmsi", "ts", "lat", "lon", "sog", "cog", "heading", "nav_status"]
STATIC_COLUMNS = [
    "mmsi",
    "vessel_name",
    "imo",
    "call_sign",
    "vessel_type",
    "length",
    "width",
    "draft",
    "cargo",
    "transceiver_class",
]


def from_noaa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=NOAA_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def vessels_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per vessel, using the most recent non-null value of each static field."""
    return df.sort_values("ts")[STATIC_COLUMNS].groupby("mmsi", as_index=False).last()
