"""Builds webapp/index.html and webapp/module.html: inlines the mammography centre list into the templates.

Run from this folder: uv run python build.py
"""

import csv
import json
from pathlib import Path


CENTERS_CSV = Path("../data/mamo_centers.csv")
PAGES = [("index.template.html", "index.html"), ("module.template.html", "module.html")]


def centers() -> list[dict]:
    rows = []
    with CENTERS_CSV.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "id": r["center_id"],
                    "name": r["name"],
                    "open": float(r["open_earliest_h"])
                    if r["open_earliest_h"]
                    else None,
                    "close": float(r["close_latest_h"])
                    if r["close_latest_h"]
                    else None,
                    "address": r["address"],
                    "city": r["city"],
                    "region": r["region"],
                    "phone": r["phone1"] or None,
                    "wait": int(r["wait_days_primary"])
                    if r["wait_days_primary"]
                    else None,
                    "updated": r["wait_updated"] or None,
                    "booking": r["online_booking_url"] or None,
                    "site": r["website"] or None,
                    "detail": r["detail_url"] or None,
                }
            )
    return rows


payload = json.dumps(centers(), ensure_ascii=False).replace("</", "<\\/")
for src, dst in PAGES:
    out = Path(dst)
    out.write_text(
        Path(src).read_text(encoding="utf-8").replace("/*CENTERS*/[]", payload),
        encoding="utf-8",
    )
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
