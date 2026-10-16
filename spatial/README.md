# Spatial enrichment

The main dataset (`data/Rakhaton2026.csv`) is not in git, so these scripts rebuild the
extra spatial data next to it. Run from this directory, in order:

```sh
uv run 01_scrape_centers.py        # data/mamo_centers.csv      (75 screening centres, GPS, waits, hours)
uv run 02_geocode_areas.py         # data/area_coordinates.csv  (point per VZP_TERRITORIAL_AREA)
uv run 03_add_spatial_features.py  # data/area_spatial_features.csv + data/Rakhaton2026_spatial.csv
```

Added columns (all `SPATIAL_*`, constant within a `VZP_TERRITORIAL_AREA`): straight-line distance
to the nearest and 2nd nearest centre, number of centres within 15/30/50 km, and booking-wait
statistics of nearby centres, and the earliest opening / latest closing hour (decimal hours,
`SPATIAL_EARLIEST_OPEN_<r>KM_H`, `SPATIAL_LATEST_CLOSE_<r>KM_H`) among centres within 30 km (the other radii were nearly identical, so dropped).
Centres without published hours are ignored; the value is empty if none in range has hours.

Limitations: the areas are districts, so there is no per-woman variation; distances are
straight-line, not travel time; the centre list and waits are a snapshot of the scraping date.
