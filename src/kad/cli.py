import argparse
import json
import logging
from datetime import date

import pandas as pd

from kad import db, ingest, schema
from kad.clean import CleanConfig, clean_positions
from kad.config import DATA_DIR, REGIONS


def clean_dir(region: str):
    return DATA_DIR / "clean" / region


def cmd_ingest(args):
    ingest.ingest_days(args.start, args.end, args.region, keep_zips=not args.delete_zips)


def cmd_clean(args):
    raw = ingest.load_raw(args.region, args.start, args.end)
    cfg = CleanConfig(gap_minutes=args.gap_minutes, max_speed_kn=args.max_speed_kn)
    positions, report = clean_positions(raw, cfg)
    report.update(region=args.region, start=str(args.start), end=str(args.end))

    out = clean_dir(args.region)
    out.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(out / "positions.parquet", index=False)
    vessels = schema.vessels_from_positions(raw[raw["mmsi"].isin(positions["mmsi"])])
    vessels.to_parquet(out / "vessels.parquet", index=False)
    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_init_db(args):
    with db.connect() as conn:
        db.init_schema(conn)


def cmd_load(args):
    src = clean_dir(args.region)
    positions = pd.read_parquet(src / "positions.parquet")
    vessels = pd.read_parquet(src / "vessels.parquet")
    with db.connect() as conn:
        db.init_schema(conn)
        db.upsert_vessels(conn, vessels)
        db.load_positions(conn, positions, args.region)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kad")
    sub = parser.add_subparsers(required=True)

    def with_range(p):
        p.add_argument("--region", choices=REGIONS, default="florida_straits")
        p.add_argument("--start", type=date.fromisoformat, required=True)
        p.add_argument("--end", type=date.fromisoformat, required=True)
        return p

    p = with_range(sub.add_parser("ingest", help="download NOAA days and filter to region"))
    p.add_argument("--delete-zips", action="store_true", help="remove zips after extraction")
    p.set_defaults(func=cmd_ingest)

    p = with_range(sub.add_parser("clean", help="validate, de-glitch, segment tracks"))
    p.add_argument("--gap-minutes", type=float, default=CleanConfig.gap_minutes)
    p.add_argument("--max-speed-kn", type=float, default=CleanConfig.max_speed_kn)
    p.set_defaults(func=cmd_clean)

    p = sub.add_parser("init-db", help="create PostGIS tables")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("load", help="load cleaned region into PostGIS")
    p.add_argument("--region", choices=REGIONS, default="florida_straits")
    p.set_defaults(func=cmd_load)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
