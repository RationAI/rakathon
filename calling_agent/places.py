"""Offline Czech place lookup using the public GeoNames country extract."""

import argparse
import hashlib
import io
import json
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from calling_agent.config import Settings


SOURCE = "https://download.geonames.org/export/dump/CZ.zip"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"


def normalize(value: str) -> str:
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKD", value.lower())
            if not unicodedata.combining(c)
        )
        .replace(",", " ")
        .split()
    )


def import_places(archive: Path, target: Path) -> int:
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        info = zipped.getinfo("CZ.txt")
        if info.file_size > 80_000_000:
            raise ValueError("Unexpected GeoNames extract size")
        rows = [
            line.split("\t") for line in zipped.read(info).decode("utf-8").splitlines()
        ]
    rows = [row for row in rows if len(row) == 19 and row[8] == "CZ"]
    regions = {row[10]: row[1] for row in rows if row[7] == "ADM1"}
    districts = {(row[10], row[11]): row[1] for row in rows if row[7] == "ADM2"}
    places = []
    for row in rows:
        if row[6] != "P" or row[7] in {"PPLQ", "PPLH", "PPLW"}:
            continue
        lat, lon = float(row[4]), float(row[5])
        if not (48 <= lat <= 52 and 12 <= lon <= 19):
            continue
        places.append(
            {
                "id": int(row[0]),
                "name": row[1],
                "aliases": sorted({row[1], row[2], *row[3].split(",")}),
                "latitude": lat,
                "longitude": lon,
                "region": regions.get(row[10], ""),
                "district": districts.get((row[10], row[11]), ""),
            }
        )
    if len(places) < 5000:
        raise ValueError("Incomplete Czech place index; previous snapshot retained")
    data = {
        "source_url": SOURCE,
        "attribution": "GeoNames",
        "license": LICENSE,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "fetched_at": datetime.now(UTC).isoformat(),
        "places": places,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return len(places)


class Gazetteer:
    def __init__(self, path: Path):
        self.path = path
        self.modified = None
        self.places = []
        self.index = {}

    def resolve(self, query: str) -> list[dict]:
        modified = self.path.stat().st_mtime_ns if self.path.exists() else None
        if modified != self.modified:
            self.places = (
                json.loads(self.path.read_text(encoding="utf-8"))["places"]
                if modified
                else []
            )
            self.index = {}
            for place in self.places:
                for alias in place["aliases"]:
                    if alias:
                        self.index.setdefault(normalize(alias), {})[place["id"]] = place
            self.modified = modified
        key = normalize(query)
        matches = list(self.index.get(key, {}).values())
        if not matches and len(key.split()) > 1:
            matches = [
                p
                for p in self.places
                if any(
                    key.startswith(normalize(a) + " ")
                    and set(key.split())
                    <= set(normalize(" ".join([a, p["district"], p["region"]])).split())
                    for a in p["aliases"]
                    if a
                )
            ]
        return [{k: v for k, v in p.items() if k != "aliases"} for p in matches]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an offline Czech place index from the public GeoNames CZ.zip"
    )
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    count = import_places(args.archive, Settings().data_dir / "places.json")
    print(f"Imported {count} Czech places from GeoNames")
