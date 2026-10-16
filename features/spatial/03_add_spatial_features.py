# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "numpy"]
# ///
"""Append centre-based spatial features to the main dataset.

Opening-hour features are decimal hours (e.g. 7.5 = 7:30).
Features are computed per VZP_TERRITORIAL_AREA (straight-line distance from the
area point to the screening centres) and joined onto every invitation. Writes
data/area_spatial_features.csv and data/Rakhaton2026_spatial.csv.

Caveats:
- the centre list is a snapshot from the scraping date, not 2020-2024 history;
- wait-time features are the current self-reported booking waits.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
EARTH_RADIUS_KM = 6371.0
RADII_KM = (15, 30, 50)
HOURS_RADIUS_KM = 30  # opening hours barely vary across radii, so only one is kept
KEY = "VZP_TERRITORIAL_AREA"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def area_features(areas: pd.DataFrame, centres: pd.DataFrame) -> pd.DataFrame:
    dist = haversine_km(
        areas["lat"].to_numpy()[:, None], areas["lon"].to_numpy()[:, None],
        centres["lat"].to_numpy()[None, :], centres["lon"].to_numpy()[None, :],
    )
    order = np.argsort(dist, axis=1)
    wait = centres["wait_days_primary"].to_numpy(dtype=float)
    out = pd.DataFrame({KEY: areas["area"]})
    out["SPATIAL_DIST_NEAREST_KM"] = np.take_along_axis(dist, order[:, :1], 1)[:, 0]
    out["SPATIAL_DIST_2ND_KM"] = np.take_along_axis(dist, order[:, 1:2], 1)[:, 0]
    for r in RADII_KM:
        out[f"SPATIAL_CENTRES_WITHIN_{r}KM"] = (dist <= r).sum(axis=1)
    out["SPATIAL_WAIT_NEAREST_DAYS"] = wait[order[:, 0]]
    out["SPATIAL_WAIT_MEAN_NEAREST3_DAYS"] = wait[order[:, :3]].mean(axis=1)
    within = np.where(dist <= 30, wait[None, :], np.nan)
    out["SPATIAL_WAIT_MIN_WITHIN_30KM_DAYS"] = np.nanmin(
        np.where(np.isnan(within), np.inf, within), axis=1
    )
    out.loc[np.isinf(out["SPATIAL_WAIT_MIN_WITHIN_30KM_DAYS"]), "SPATIAL_WAIT_MIN_WITHIN_30KM_DAYS"] = np.nan
    # earliest opening / latest closing hour (decimal) among centres within HOURS_RADIUS_KM
    open_h = centres["open_earliest_h"].to_numpy(dtype=float)
    close_h = centres["close_latest_h"].to_numpy(dtype=float)
    r = HOURS_RADIUS_KM
    in_range = dist <= r
    out[f"SPATIAL_EARLIEST_OPEN_{r}KM_H"] = np.where(in_range, open_h[None, :], np.nan).min(axis=1, initial=np.inf, where=in_range & ~np.isnan(open_h)[None, :])
    out[f"SPATIAL_LATEST_CLOSE_{r}KM_H"] = np.where(in_range, close_h[None, :], np.nan).max(axis=1, initial=-np.inf, where=in_range & ~np.isnan(close_h)[None, :])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def main() -> None:
    areas = pd.read_csv(DATA / "area_coordinates.csv")
    centres = pd.read_csv(DATA / "mamo_centers.csv").dropna(subset=["lat", "lon", "wait_days_primary"])
    feats = area_features(areas, centres)
    feats.to_csv(DATA / "area_spatial_features.csv", index=False)

    df = pd.read_csv(DATA / "Rakhaton2026.csv", dtype={"PERSON_ID": str, "INVITATION_ID": str})
    n = len(df)
    merged = df.merge(feats, on=KEY, how="left", validate="many_to_one")
    assert len(merged) == n, "merge changed the row count"
    missing = merged["SPATIAL_DIST_NEAREST_KM"].isna().sum()
    assert missing == 0, f"{missing} rows without spatial features"
    merged.to_csv(DATA / "Rakhaton2026_spatial.csv", index=False)
    print(f"Wrote {len(feats)} area feature rows and {n:,} enriched invitations")
    print(feats.describe().round(1).T)


if __name__ == "__main__":
    main()
