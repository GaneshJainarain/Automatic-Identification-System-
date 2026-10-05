import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between arrays of points (degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
