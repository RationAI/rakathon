# /// script
# requires-python = ">=3.12"
# ///
"""Geocode the VZP_TERRITORIAL_AREA values to a representative point.

The area names are district (okres) names, so each is geocoded as the centre of
the OSM district boundary via Nominatim (1 request/s, results cached).
Writes data/area_coordinates.csv.
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
DATASET = DATA / "Rakhaton2026.csv"
OUT = DATA / "area_coordinates.csv"

# Names that are not a plain "okres <name>" in OSM
QUERY_OVERRIDES = {
    "PRAHA-HL.MĚSTO": "Praha, Česko",
    "BRNO-MĚSTO": "Brno, Česko",
    "PLZEŇ-MĚSTO": "Plzeň, Česko",
    "OSTRAVA": "Ostrava, Česko",
}
LAT_RANGE, LON_RANGE = (48.5, 51.1), (12.0, 18.9)  # sanity box around Czechia


def areas() -> list[str]:
    with DATASET.open(encoding="utf-8") as f:
        return sorted({row["VZP_TERRITORIAL_AREA"] for row in csv.DictReader(f)})


def geocode(query: str) -> tuple[float, float, str]:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "cz"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "rakathon-mamma-research/1.0"})
    hits = json.load(urllib.request.urlopen(req, timeout=30))
    if not hits:
        raise LookupError(query)
    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0]["display_name"]


def main() -> None:
    cache = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            cache = {r["area"]: r for r in csv.DictReader(f)}

    rows, failed = [], []
    for area in areas():
        if area in cache:
            rows.append(cache[area])
            continue
        query = QUERY_OVERRIDES.get(area, f"okres {area.title()}, Česko")
        try:
            lat, lon, name = geocode(query)
            assert LAT_RANGE[0] < lat < LAT_RANGE[1] and LON_RANGE[0] < lon < LON_RANGE[1]
            rows.append({"area": area, "lat": lat, "lon": lon, "matched": name})
        except (LookupError, AssertionError) as e:
            failed.append(area)
            print(f"FAILED {area}: {e!r}")
        time.sleep(1.1)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["area", "lat", "lon", "matched"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} areas to {OUT}; {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
