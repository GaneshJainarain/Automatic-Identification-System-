import numpy as np
import pandas as pd
import pytest

from kad.clean import CleanConfig, clean_positions, find_spikes, valid_mmsi
from kad.geo import haversine_km

T0 = pd.Timestamp("2023-01-01", tz="UTC")


def track(mmsi, n, lat0=25.0, lon0=-80.0, step_s=60, dlat=0.001):
    """Straight northbound track at ~3.6 kn."""
    return pd.DataFrame(
        {
            "mmsi": mmsi,
            "ts": [T0 + pd.Timedelta(seconds=i * step_s) for i in range(n)],
            "lat": lat0 + dlat * np.arange(n),
            "lon": np.full(n, lon0),
            "sog": np.float32(3.6),
            "cog": np.float32(0.0),
            "heading": np.float32(0.0),
            "nav_status": 0,
        }
    )


def test_haversine_one_degree_latitude():
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.01)


def test_valid_mmsi():
    s = pd.Series([366999999, 123456789, 3669999, 993660001, 111366001])
    assert valid_mmsi(s).tolist() == [True, False, False, False, False]


def test_drops_invalid_mmsi_coords_and_duplicates():
    df = pd.concat([track(366000001, 20), track(123456789, 20)], ignore_index=True)
    df.loc[3, ["lat", "lon"]] = 0.0
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)

    out, report = clean_positions(df)

    assert report["dropped"]["invalid_mmsi"] == 20
    assert report["dropped"]["invalid_coords"] == 1
    assert report["dropped"]["duplicate_timestamp"] == 1
    assert len(out) == 19
    assert out["mmsi"].unique().tolist() == [366000001]


def test_sentinels_become_null():
    df = track(366000001, 20)
    df.loc[0, "sog"] = 102.3
    df.loc[1, "cog"] = 360.0
    df.loc[2, "heading"] = 511.0
    out, _ = clean_positions(df)
    assert out[["sog", "cog", "heading"]].isna().sum().tolist() == [1, 1, 1]


def test_single_point_glitch_removed_but_teleport_kept():
    glitch = track(366000001, 20)
    glitch.loc[10, "lat"] += 1.0  # out and back: GPS glitch
    teleport = track(366000002, 20)
    teleport.loc[10:, "lat"] += 1.0  # jumps and stays: spoofing signal, keep

    df = pd.concat([glitch, teleport], ignore_index=True).sort_values(["mmsi", "ts"])
    spikes = find_spikes(df.reset_index(drop=True), CleanConfig())
    assert spikes.sum() == 1

    out, report = clean_positions(df)
    assert report["dropped"]["position_spike"] == 1
    assert (out["mmsi"] == 366000002).sum() == 20


def test_segments_split_on_gap_and_short_segments_dropped():
    a = track(366000001, 15)
    b = track(366000001, 15, lat0=25.2)
    b["ts"] += pd.Timedelta(hours=2)
    c = track(366000001, 5, lat0=25.4)
    c["ts"] += pd.Timedelta(hours=5)

    out, report = clean_positions(pd.concat([a, b, c], ignore_index=True))

    assert report["segments"] == 2
    assert report["dropped"]["short_segment"] == 5
    first_of_b = out[out["segment_id"] == 1].iloc[0]
    assert first_of_b["gap_before_s"] == pytest.approx(2 * 3600 - 14 * 60)
