"""Generate the figures used in README.md from the cleaned Florida Straits data.

Usage: python scripts/make_readme_figures.py   (after `kad ingest` + `kad clean`)
"""

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from kad import schema
from kad.clean import CleanConfig, find_spikes, valid_coords, valid_mmsi
from kad.config import DATA_DIR, REGIONS
from kad.geo import haversine_km

REGION = "florida_straits"
OUT = Path("docs/img")

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e5e4df"
SURFACE = "#ffffff"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    }
)

PLACES = {
    "Miami": (25.77, -80.19),
    "Fort Lauderdale": (26.12, -80.14),
    "Key West": (24.56, -81.78),
    "Havana": (23.13, -82.38),
    "Bimini": (25.73, -79.27),
}


def titled(ax, title, subtitle):
    ax.set_title(title, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def traffic_density(pos):
    bbox = REGIONS[REGION]
    fig, ax = plt.subplots(figsize=(8, 7.2))
    hb = ax.hexbin(
        pos["lon"],
        pos["lat"],
        gridsize=180,
        bins=None,
        norm=LogNorm(),
        cmap=BLUES,
        mincnt=1,
        linewidths=0,
        extent=(bbox.lon_min, bbox.lon_max, bbox.lat_min, bbox.lat_max),
    )
    for name, (lat, lon) in PLACES.items():
        ax.plot(lon, lat, "o", ms=5, color=INK, mec=SURFACE, mew=1.5)
        near_edge = lon > bbox.lon_max - 0.5
        ax.annotate(
            name,
            (lon, lat),
            xytext=(0, -9) if near_edge else (6, 4),
            textcoords="offset points",
            ha="center" if near_edge else "left",
            va="top" if near_edge else "baseline",
            color=INK,
            fontsize=9,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)],
        )
    ax.set_xlim(bbox.lon_min, bbox.lon_max)
    ax.set_ylim(bbox.lat_min, bbox.lat_max)
    ax.set_aspect(1 / np.cos(np.radians(25)))
    ax.grid(False)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    titled(
        ax,
        "One day of vessel traffic in the Florida Straits",
        f"{len(pos):,} cleaned AIS positions · {pos['mmsi'].nunique():,} vessels · "
        "2023-01-01 · NOAA MarineCadastre",
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Positions per cell (log scale)")
    cb.outline.set_visible(False)
    save(fig, "traffic_density.png")


def cleaning_funnel(report):
    labels = {
        "invalid_mmsi": "Invalid vessel ID (MMSI)",
        "invalid_coords": "Invalid coordinates",
        "duplicate_timestamp": "Duplicate timestamps",
        "position_spike": "GPS glitches (jump out and back)",
        "short_segment": "Tracks with < 10 fixes",
    }
    rows = [(labels[k], v) for k, v in report["dropped"].items()]
    names, counts = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, counts, height=0.6, color=BLUE)
    for yi, c in zip(y, counts):
        ax.annotate(
            f"{c:,}",
            (c, yi),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Rows removed")
    ax.set_xlim(0, max(counts) * 1.15)
    kept = report["output_rows"]
    titled(
        ax,
        "What cleaning removed",
        f"{report['input_rows']:,} raw rows → {kept:,} kept ({kept / report['input_rows']:.1%})",
    )
    save(fig, "cleaning_funnel.png")


def vessel_mix(vessels):
    counts = schema.vessel_category(vessels["vessel_type"]).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(counts.index, counts.values, height=0.6, color=BLUE)
    for i, c in enumerate(counts.values):
        ax.annotate(
            f"{c:,}",
            (c, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, counts.max() * 1.12)
    ax.set_xlabel("Vessels")
    titled(
        ax,
        "Who is out there: vessels by type",
        "Recreational boats dominate — the small-craft traffic border enforcement cares about",
    )
    save(fig, "vessel_mix.png")


def gap_distribution(pos, cfg):
    gaps_min = pos["gap_before_s"].dropna() / 60
    gaps_min = gaps_min[gaps_min > 0]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(np.log10(gaps_min.min()), np.log10(gaps_min.max()), 70)
    ax.hist(gaps_min, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(cfg.gap_minutes, color=ORANGE, lw=2)
    n_long = int((gaps_min > cfg.gap_minutes).sum())
    ax.annotate(
        f"{cfg.gap_minutes:.0f}-minute split\n{n_long:,} gaps longer → new track,\n"
        "candidates for “going dark”",
        (cfg.gap_minutes, ax.get_ylim()[1] * 0.3),
        xytext=(8, 0),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="top",
    )
    ax.set_xlabel("Time since vessel's previous report (minutes, log scale)")
    ax.set_ylabel("Reports (log scale)")
    ax.set_title("Reporting gaps: most vessels report every few seconds to minutes")
    ax.grid(axis="x", visible=False)
    save(fig, "gap_distribution.png")


def glitch_example(raw, mmsi, window=40):
    df = raw[valid_mmsi(raw["mmsi"]) & valid_coords(raw)]
    df = df[df["mmsi"] == mmsi].sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    spikes = find_spikes(df, CleanConfig())
    i = int(np.flatnonzero(spikes)[0])
    df, spikes = df.iloc[i - window : i + window], spikes.iloc[i - window : i + window]
    glitch = df[spikes].iloc[0]
    before, after = df.loc[i - 1], df.loc[i + 1]
    dist = haversine_km(before["lat"], before["lon"], glitch["lat"], glitch["lon"])
    minutes = (after["ts"] - before["ts"]).total_seconds() / 60

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(df["lon"], df["lat"], "-", color=MUTED, lw=1, zorder=1, label="Raw reported path")
    kept = df[~spikes]
    ax.plot(kept["lon"], kept["lat"], "o", ms=3, color=BLUE, zorder=2, label="Kept fixes")
    ax.plot(
        glitch["lon"],
        glitch["lat"],
        "o",
        ms=9,
        color=ORANGE,
        mec=SURFACE,
        mew=1.5,
        zorder=3,
        label="Removed glitch",
    )
    ax.annotate(
        f"Single fix {dist:.0f} km off-track,\nback on track {minutes:.0f} min later\n"
        f"(implied speed > {CleanConfig().max_speed_kn:.0f} kn both ways)",
        (glitch["lon"], glitch["lat"]),
        xytext=(-14, 0),
        textcoords="offset points",
        color=INK,
        fontsize=9,
        va="center",
        ha="right",
    )
    ax.set_aspect(1 / np.cos(np.radians(df["lat"].mean())))
    ax.margins(0.25, 0.15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    titled(ax, "A real GPS glitch, removed", f"Vessel {mmsi} · 2023-01-01")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "glitch_example.png")


def anomaly_types():
    """Illustrative (synthetic) sketches of the behaviors the detectors target."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
    t = np.linspace(0, 1, 60)

    ax = axes[0]
    th = np.linspace(0, 6 * np.pi, 60)
    x = np.r_[
        np.linspace(0, 3, 25), 4 - np.cos(th) + rng.normal(0, 0.08, 60), np.linspace(3, 8, 25)
    ]
    y = np.r_[np.zeros(25), np.sin(th) + rng.normal(0, 0.08, 60), np.linspace(0, 3, 25)]
    ax.plot(x, y, color=BLUE, lw=2)
    ax.plot(x[24:86], y[24:86], color=ORANGE, lw=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Loitering", fontsize=11)
    ax.text(
        0.5,
        -0.12,
        "Slow circling far from port",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK_2,
        fontsize=9,
    )

    ax = axes[1]
    ax.plot(t[:25] * 8, t[:25] * 2, color=BLUE, lw=2)
    ax.plot(t[40:] * 8, t[40:] * 2 + 0.6, color=BLUE, lw=2)
    ax.plot(
        [t[24] * 8, t[40] * 8], [t[24] * 2, t[40] * 2 + 0.6], ls=(0, (3, 3)), color=ORANGE, lw=2
    )
    ax.annotate(
        "no AIS",
        ((t[24] + t[40]) * 4 - 0.4, (t[24] + t[40]) + 0.3),
        color=ORANGE,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("Going dark", fontsize=11)
    ax.text(
        0.5,
        -0.12,
        "Transponder off, reappears elsewhere",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK_2,
        fontsize=9,
    )

    ax = axes[2]
    ax.plot(np.linspace(0, 4, 30), np.linspace(0, 2, 30), color=BLUE, lw=2)
    ax.plot(np.linspace(8, 4.15, 30), np.linspace(0, 2, 30), color="#1baf7a", lw=2)
    ax.plot([4], [2], "o", ms=16, mfc="none", mec=ORANGE, mew=2)
    ax.set_title("Rendezvous", fontsize=11)
    ax.text(
        0.5,
        -0.12,
        "Two vessels meet, slow, part",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK_2,
        fontsize=9,
    )

    ax = axes[3]
    ax.plot(t[:30] * 8, t[:30] * 0.5, color=BLUE, lw=2)
    ax.plot(t[30:] * 8, t[30:] * 0.5 + 2, color=BLUE, lw=2)
    ax.annotate(
        "",
        (t[30] * 8, t[30] * 0.5 + 2),
        (t[29] * 8, t[29] * 0.5),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 2},
    )
    ax.set_title("Spoofed jump", fontsize=11)
    ax.text(
        0.5,
        -0.12,
        "Physically impossible teleport",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK_2,
        fontsize=9,
    )

    ax = axes[4]
    for k in range(6):
        ax.plot(t * 8, 0.3 * np.sin(t * 3) + rng.normal(0, 0.06) + 0.05 * k, color=GRID, lw=2)
    dev = 0.3 * np.sin(t * 3) + np.where(
        t > 0.35, 2.2 * np.sin(np.clip((t - 0.35) * 4, 0, np.pi)), 0
    )
    ax.plot(t * 8, dev, color=ORANGE, lw=2)
    ax.set_title("Route deviation", fontsize=11)
    ax.text(
        0.5,
        -0.12,
        "Leaves the usual shipping lane",
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=INK_2,
        fontsize=9,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
        ax.margins(0.15)
    fig.suptitle(
        "Behaviors the detectors target (illustrative)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=12,
    )
    fig.tight_layout()
    save(fig, "anomaly_types.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = DATA_DIR / "clean" / REGION
    pos = pd.read_parquet(clean / "positions.parquet")
    vessels = pd.read_parquet(clean / "vessels.parquet")
    report = json.loads((clean / "cleaning_report.json").read_text())
    raw = pd.read_parquet(DATA_DIR / "raw" / REGION / f"{report['start']}.parquet")

    traffic_density(pos)
    cleaning_funnel(report)
    vessel_mix(vessels)
    gap_distribution(pos, CleanConfig(**report["config"]))
    glitch_example(raw, mmsi=367468230)
    anomaly_types()


if __name__ == "__main__":
    main()
