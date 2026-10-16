# /// script
# requires-python = ">=3.12"
# ///
"""Scrape the list of mammography screening centres from mamo.cz.

The page embeds all centres as JSON (with GPS coordinates, opening hours and
the self-reported booking wait). Writes data/mamo_centers.csv.

Note: waiting time is a snapshot from the day of scraping, not a history.
"""
import csv
import html
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

URL = "https://www.mamo.cz/centra/"
OUT = Path(__file__).resolve().parents[1] / "data" / "mamo_centers.csv"


def clean(fragment: str | None) -> str:
    if not fragment:
        return ""
    text = re.sub(r"</li>|<br\s*/?>|</p>", "; ", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", "", text)).replace("\xa0", " ")
    return re.sub(r"\s*;\s*(;\s*)*", " | ", text).strip(" |")


TIME_RANGE = re.compile(r"(\d{1,2})[:.](\d{2})\s*[–‐-]\s*(\d{1,2})[:.](\d{2})")


def hour_span(hours: str) -> tuple[float | None, float | None]:
    """Earliest opening and latest closing time (decimal hours) over all listed ranges.

    Segments about phone ordering are skipped; centres without hours give (None, None).
    """
    opens, closes = [], []
    for segment in hours.split("|"):
        if "telefon" in segment.lower():
            continue
        for h1, m1, h2, m2 in TIME_RANGE.findall(segment):
            opens.append(int(h1) + int(m1) / 60)
            closes.append(int(h2) + int(m2) / 60)
    return (min(opens), max(closes)) if opens else (None, None)


def main() -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    page = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")

    regions = dict(re.findall(r'<option value="(\d+)">([^<]+)</option>', page))
    first = re.search(r'\{"id":\d+,"title"', page)
    start = page.rfind("[", 0, first.start())
    centres, _ = json.JSONDecoder().raw_decode(page[start:])

    rows = []
    for c in centres:
        modified = (c.get("appointPeriodLastmodAt") or {}).get("date") or ""
        hours = clean(c["officeTime"])
        open_h, close_h = hour_span(hours)
        rows.append({
            "center_id": c["id"],
            "name": c["title"],
            "address": c["address"],
            "city": c["city"],
            "postal_code": c["postalcode"],
            "region": regions.get(c["region"], c["region"]),
            "lat": c["gpsLat"],
            "lon": c["gpsLon"],
            "wait_days_primary": c["appointPeriod"],
            "wait_days_secondary": c["appointPeriod2"],
            "wait_updated": modified[:10],
            "opening_hours": hours,
            "open_earliest_h": open_h,
            "close_latest_h": close_h,
            "phone1": c["phone1"],
            "phone2": c["phone2"],
            "email": c["email"],
            "website": c["url"],
            "online_booking_url": c["appointUrl"],
            "contact_info": clean(c["contact"]),
            "comment": clean(c["comment"]),
            "detail_url": f"https://www.mamo.cz/centrum/{c['id']}-{c['slug']}/",
            "scraped": date.today().isoformat(),
        })

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} centres to {OUT}")


if __name__ == "__main__":
    main()
