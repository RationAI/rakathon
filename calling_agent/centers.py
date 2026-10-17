"""Public mamo.cz directory; published waits are never appointment availability."""

import argparse
import hashlib
import html
import json
import re
from datetime import UTC, datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import httpx

from calling_agent.config import Settings
from calling_agent.places import Gazetteer, normalize
from calling_agent.schemas import Center


SOURCE = "https://www.mamo.cz/centra/"


def clean(value: str | None) -> str:
    return re.sub(
        r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    ).strip()


def safe_url(value: str | None) -> str:
    value = (value or "").strip()
    return value if value.startswith(("https://", "http://")) else ""


def parse_directory(page: str) -> list[Center]:
    regions = dict(re.findall(r'<option value="(\d+)">([^<]+)</option>', page))
    first = re.search(r'\{"id":\d+,"title"', page)
    if first is None:
        raise ValueError(
            "mamo.cz directory structure changed; previous snapshot retained"
        )
    start = page.rfind("[", 0, first.start())
    records, _ = json.JSONDecoder().raw_decode(page[start:])
    centers = []
    for item in records:
        modified = (item.get("appointPeriodLastmodAt") or {}).get("date")
        centers.append(
            Center(
                id=item["id"],
                name=clean(item["title"]),
                city=clean(item["city"]),
                address=clean(item["address"]),
                postal_code=item["postalcode"],
                region=clean(regions.get(str(item["region"]), str(item["region"]))),
                phone=[clean(item[k]) for k in ("phone1", "phone2") if item.get(k)],
                email=item.get("appointEmail") or item.get("email") or "",
                website=safe_url(item.get("url")),
                online_booking_url=safe_url(item.get("appointUrl")),
                opening_hours=clean(item.get("officeTime")),
                reported_wait_days=item.get("appointPeriod"),
                wait_updated=modified[:10] if modified else None,
                latitude=float(item["gpsLat"]),
                longitude=float(item["gpsLon"]),
                source_url=f"https://www.mamo.cz/centrum/{item['id']}-{item['slug']}/",
            )
        )
    if len(centers) < 50 or len({c.id for c in centers}) != len(centers):
        raise ValueError(
            "Incomplete or duplicated mamo.cz directory; previous snapshot retained"
        )
    return centers


def write_snapshot(page: str, target: Path) -> int:
    centers = parse_directory(page)
    snapshot = {
        "source_url": SOURCE,
        "fetched_at": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(page.encode()).hexdigest(),
        "availability": "unknown",
        "centers": [c.model_dump() for c in centers],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)
    return len(centers)


class Directory:
    def __init__(self, path: Path):
        self.path = path
        self.places = Gazetteer(path.with_name("places.json"))

    def snapshot(self) -> dict:
        if not self.path.exists():
            return {
                "source_url": SOURCE,
                "fetched_at": None,
                "availability": "unknown",
                "centers": [],
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, center_id: int) -> dict:
        for center in self.snapshot()["centers"]:
            if center["id"] == center_id:
                return center
        raise ValueError("Centrum není v ověřeném adresáři mamo.cz")

    def search(self, query: str) -> dict:
        snapshot = self.snapshot()
        locations = self.places.resolve(query)
        if len(locations) > 1:
            return {
                "centers": [],
                "matches": 0,
                "availability": "unknown",
                "ranking": "location ambiguous",
                "needs_clarification": True,
                "location_candidates": locations[:8],
                "source_url": SOURCE,
                "fetched_at": snapshot["fetched_at"],
                "note": "Tento název má více míst. Zeptejte se na okres nebo blízké větší město; nehádejte polohu.",
            }
        if len(locations) == 1:
            location = locations[0]
            lat, lon = radians(location["latitude"]), radians(location["longitude"])
            ranked = []
            for center in snapshot["centers"]:
                clat, clon = radians(center["latitude"]), radians(center["longitude"])
                distance = (
                    6371.0088
                    * 2
                    * asin(
                        min(
                            1.0,
                            sqrt(
                                sin((clat - lat) / 2) ** 2
                                + cos(lat) * cos(clat) * sin((clon - lon) / 2) ** 2
                            ),
                        )
                    )
                )
                ranked.append((distance, center))
            ranked.sort(key=lambda item: (item[0], item[1]["id"]))
            return {
                "centers": [{**c, "distance_km": round(d, 1)} for d, c in ranked[:3]],
                "matches": len(ranked),
                "origin": location,
                "location_source": "https://www.geonames.org/",
                "ranking": "straight_line_distance",
                "availability": "unknown",
                "source_url": SOURCE,
                "fetched_at": snapshot["fetched_at"],
                "note": "Vzdálenost je orientační vzdušnou čarou od středu uvedené obce, nikoli dojezd ani přesná adresa. Termíny pouze simulujeme.",
            }
        tokens = normalize(query).split()
        scored = []
        for center in snapshot["centers"]:
            city = normalize(center["city"])
            text = normalize(
                " ".join(center[k] for k in ("name", "city", "region", "address"))
            )
            if all(token in text for token in tokens):
                score = 10 * (normalize(query) == city) + sum(
                    token in city for token in tokens
                )
                scored.append((score, center))
        scored.sort(key=lambda row: (-row[0], row[1]["name"]))
        return {
            "centers": [c for _, c in scored[:6]],
            "matches": len(scored),
            "source_url": SOURCE,
            "fetched_at": snapshot["fetched_at"],
            "availability": "unknown",
            "ranking": "text relevance, not distance or travel time",
            "note": "Objednací doba je publikovaný orientační údaj, nikoli volný termín. Bez shody se zeptejte na kraj nebo preferované centrum.",
        }


def main():
    parser = argparse.ArgumentParser(
        description="Refresh the public mamo.cz center directory"
    )
    parser.add_argument(
        "--html", type=Path, help="Import a previously downloaded mamo.cz HTML page"
    )
    args = parser.parse_args()
    if args.html:
        page = args.html.read_text(encoding="utf-8")
    else:
        response = httpx.get(
            SOURCE,
            timeout=45,
            follow_redirects=True,
            headers={"User-Agent": "RakathonMamoDirectory/0.1"},
        )
        response.raise_for_status()
        page = response.text
    count = write_snapshot(page, Settings().data_dir / "centers.json")
    print(f"Imported {count} public centers from {SOURCE}")


if __name__ == "__main__":
    main()
