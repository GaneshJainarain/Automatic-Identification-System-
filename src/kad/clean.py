"""Validate, de-glitch, and segment AIS position reports.

Every rule records how many rows it removed so data quality is auditable.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from kad.geo import KM_PER_NM, haversine_km

# AIS "not available" sentinels (ITU-R M.1371).
SOG_NA = 102.3
COG_NA = 360.0
HEADING_NA = 511.0


@dataclass
class CleanConfig:
    # Implied speed between consecutive fixes above this is implausible for surface vessels.
    max_speed_kn: float = 50.0
    # Ignore implied-speed spikes over tiny distances (timestamp jitter between receivers).
    min_jump_km: float = 1.0
    # Split a vessel's history into separate tracks on transmission gaps longer than this.
    gap_minutes: float = 30.0
    # Drop track segments with fewer fixes than this.
    min_points: int = 10
    # Spike removal repeats so back-to-back glitches are caught.
    spike_passes: int = 3


def valid_mmsi(mmsi: pd.Series) -> pd.Series:
    """Ship-station MMSIs: 9 digits with a Maritime Identification Digit of 201-775.

    Excludes base stations (00...), SAR aircraft (111...), AtoNs (99...), SARTs (970...)
    and obviously fake IDs like 123456789 that fall outside the MID table.
    """
    mid = mmsi // 1_000_000
    return mmsi.between(100_000_000, 999_999_999) & mid.between(201, 775)


def valid_coords(df: pd.DataFrame) -> pd.Series:
    return (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & ~((df["lat"] == 0) & (df["lon"] == 0))
    )


def null_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sog"] = df["sog"].mask((df["sog"] >= SOG_NA) | (df["sog"] < 0))
    df["cog"] = df["cog"].mask((df["cog"] >= COG_NA) | (df["cog"] < 0))
    df["heading"] = df["heading"].mask((df["heading"] >= 360) | (df["heading"] < 0))
    return df


def step_kinematics(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distance (km), elapsed time (s) and implied speed (kn) from each fix's predecessor.

    Expects df sorted by (mmsi, ts). Values are NaN for each vessel's first fix.
    """
    same = df["mmsi"].to_numpy()[1:] == df["mmsi"].to_numpy()[:-1]
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    ts = df["ts"].to_numpy()

    dist = np.full(len(df), np.nan)
    dt = np.full(len(df), np.nan)
    dist[1:] = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
    dist[1:][~same] = np.nan
    dt[1:][~same] = np.nan

    speed = dist / np.maximum(dt, 1.0) * 3600 / KM_PER_NM
    return dist, dt, speed


def find_spikes(df: pd.DataFrame, cfg: CleanConfig) -> pd.Series:
    """Single-fix GPS glitches: an implausible jump away *and* an implausible jump back.

    A sustained teleport (jump with no return) is not removed here — that pattern is a
    spoofing signal for the detectors, not a cleaning problem.
    """
    dist, _, speed = step_kinematics(df)
    jump_in = (speed > cfg.max_speed_kn) & (dist > cfg.min_jump_km)
    jump_out = np.zeros_like(jump_in)
    jump_out[:-1] = jump_in[1:]
    return pd.Series(jump_in & jump_out, index=df.index)


def segment_tracks(df: pd.DataFrame, cfg: CleanConfig) -> pd.DataFrame:
    """Assign segment_id (new track on vessel change or long gap) and gap_before_s."""
    df = df.copy()
    _, dt, _ = step_kinematics(df)
    new_vessel = np.isnan(dt)
    new_segment = new_vessel | (dt > cfg.gap_minutes * 60)
    df["segment_id"] = np.cumsum(new_segment) - 1
    df["gap_before_s"] = dt
    return df


def clean_positions(df: pd.DataFrame, cfg: CleanConfig | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or CleanConfig()
    report: dict = {"config": asdict(cfg), "input_rows": len(df), "dropped": {}}

    def drop(frame: pd.DataFrame, mask: pd.Series, rule: str) -> pd.DataFrame:
        report["dropped"][rule] = report["dropped"].get(rule, 0) + int(mask.sum())
        return frame[~mask]

    df = drop(df, ~valid_mmsi(df["mmsi"]), "invalid_mmsi")
    df = drop(df, ~valid_coords(df), "invalid_coords")
    df = null_sentinels(df)
    df = df.sort_values(["mmsi", "ts"], kind="stable")
    df = drop(df, df.duplicated(["mmsi", "ts"]), "duplicate_timestamp")

    for _ in range(cfg.spike_passes):
        spikes = find_spikes(df, cfg)
        if not spikes.any():
            break
        df = drop(df, spikes, "position_spike")

    df = segment_tracks(df.reset_index(drop=True), cfg)
    sizes = df.groupby("segment_id")["segment_id"].transform("size")
    df = drop(df, sizes < cfg.min_points, "short_segment")
    # Renumber so segment ids are dense after short segments are dropped.
    df["segment_id"] = pd.factorize(df["segment_id"])[0]

    report.update(
        output_rows=len(df),
        vessels=int(df["mmsi"].nunique()),
        segments=int(df["segment_id"].nunique()),
    )
    return df.reset_index(drop=True), report
