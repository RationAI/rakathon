import json
from pathlib import Path


def directory_fixture(folder: Path):
    # Synthetic center records. The full public directory is imported separately.
    centers = [
        {
            "id": 1,
            "name": "Testovací centrum Brno",
            "city": "Brno",
            "address": "Testovací 1",
            "postal_code": "602 00",
            "region": "Jihomoravský kraj",
            "phone": ["+420 555 000 001"],
            "email": "",
            "website": "",
            "online_booking_url": "",
            "opening_hours": "po-pá",
            "reported_wait_days": 12,
            "wait_updated": "2026-10-01",
            "latitude": 49.19,
            "longitude": 16.60,
            "source_url": "https://www.mamo.cz/centrum/1-test/",
        },
        {
            "id": 2,
            "name": "Testovací centrum Třebíč",
            "city": "Třebíč",
            "address": "Testovací 2",
            "postal_code": "674 01",
            "region": "Kraj Vysočina",
            "phone": [],
            "email": "",
            "website": "",
            "online_booking_url": "",
            "opening_hours": "",
            "reported_wait_days": None,
            "wait_updated": None,
            "latitude": 49.21,
            "longitude": 15.88,
            "source_url": "https://www.mamo.cz/centrum/2-test/",
        },
    ]
    (folder / "centers.json").write_text(
        json.dumps(
            {
                "source_url": "https://www.mamo.cz/centra/",
                "fetched_at": "2026-10-17T08:00:00Z",
                "centers": centers,
            }
        ),
        encoding="utf-8",
    )
