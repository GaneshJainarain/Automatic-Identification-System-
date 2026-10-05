import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("KAD_DATA_DIR", "data"))
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://kad:kad@localhost:5432/kad")


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat, lon) -> np.ndarray:
        return (
            (lat >= self.lat_min)
            & (lat <= self.lat_max)
            & (lon >= self.lon_min)
            & (lon <= self.lon_max)
        )


REGIONS = {
    # Florida Straits / South Florida approaches. Coverage thins toward Cuba because
    # NOAA's receivers are US shore stations.
    "florida_straits": BBox(lat_min=23.0, lat_max=27.5, lon_min=-83.5, lon_max=-79.0),
    "gulf_of_mexico": BBox(lat_min=18.0, lat_max=31.0, lon_min=-98.0, lon_max=-80.5),
    "southern_california": BBox(lat_min=31.5, lat_max=34.5, lon_min=-121.0, lon_max=-116.5),
}
