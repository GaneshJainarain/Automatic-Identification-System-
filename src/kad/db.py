"""Load cleaned positions and vessel metadata into PostgreSQL/PostGIS."""

import logging
from pathlib import Path

import pandas as pd
import psycopg

from kad.config import DATABASE_URL

log = logging.getLogger(__name__)

SCHEMA_SQL = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

POSITION_DB_COLUMNS = [
    "region",
    "mmsi",
    "ts",
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "nav_status",
    "segment_id",
    "gap_before_s",
]
VESSEL_DB_COLUMNS = [
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


def connect(url: str = DATABASE_URL) -> psycopg.Connection:
    return psycopg.connect(url)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_SQL.read_text())
    conn.commit()


def _copy(cur: psycopg.Cursor, table: str, df: pd.DataFrame, columns: list[str]) -> None:
    # COPY needs None for NULL; NaN / pd.NA would otherwise be sent as values.
    rows = df[columns].astype(object).where(df[columns].notna(), None)
    with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as copy:
        for row in rows.itertuples(index=False, name=None):
            copy.write_row(row)


def load_positions(conn: psycopg.Connection, df: pd.DataFrame, region: str) -> None:
    """Replace all positions for region with df."""
    df = df.assign(region=region)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE region = %s", (region,))
        _copy(cur, "positions", df, POSITION_DB_COLUMNS)
    conn.commit()
    log.info("loaded %d positions for %s", len(df), region)


def upsert_vessels(conn: psycopg.Connection, df: pd.DataFrame) -> None:
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in VESSEL_DB_COLUMNS[1:])
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE vessels_stage (LIKE vessels) ON COMMIT DROP")
        _copy(cur, "vessels_stage", df, VESSEL_DB_COLUMNS)
        cur.execute(
            f"INSERT INTO vessels SELECT * FROM vessels_stage "
            f"ON CONFLICT (mmsi) DO UPDATE SET {updates}"
        )
    conn.commit()
    log.info("upserted %d vessels", len(df))
