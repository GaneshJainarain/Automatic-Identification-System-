CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS vessels (
    mmsi              bigint PRIMARY KEY,
    vessel_name       text,
    imo               text,
    call_sign         text,
    vessel_type       smallint,
    length            real,
    width             real,
    draft             real,
    cargo             smallint,
    transceiver_class text
);

CREATE TABLE IF NOT EXISTS positions (
    region       text             NOT NULL,
    mmsi         bigint           NOT NULL,
    ts           timestamptz      NOT NULL,
    lat          double precision NOT NULL,
    lon          double precision NOT NULL,
    sog          real,
    cog          real,
    heading      real,
    nav_status   smallint,
    segment_id   integer          NOT NULL,
    gap_before_s real,
    geom geometry(Point, 4326)
        GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED,
    PRIMARY KEY (region, mmsi, ts)
);

CREATE INDEX IF NOT EXISTS positions_geom_idx ON positions USING gist (geom);
CREATE INDEX IF NOT EXISTS positions_ts_idx ON positions (ts);
CREATE INDEX IF NOT EXISTS positions_segment_idx ON positions (region, segment_id);
