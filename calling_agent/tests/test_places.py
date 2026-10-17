import json
import tempfile
import unittest
from pathlib import Path

from calling_agent.centers import Directory
from calling_agent.tests.helpers import directory_fixture


class NearbyCentersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        directory_fixture(self.folder)
        self.path = self.folder / "places.json"
        self.path.write_text(
            json.dumps(
                {
                    "places": [
                        {
                            "id": 10,
                            "name": "Testovací Ves",
                            "aliases": ["Testovací Ves"],
                            "latitude": 49.22,
                            "longitude": 15.89,
                            "district": "Okres Třebíč",
                            "region": "Kraj Vysočina",
                        },
                        {
                            "id": 11,
                            "name": "Nová Ves",
                            "aliases": ["Nová Ves"],
                            "latitude": 49.22,
                            "longitude": 15.89,
                            "district": "Okres Třebíč",
                            "region": "Kraj Vysočina",
                        },
                        {
                            "id": 12,
                            "name": "Nová Ves",
                            "aliases": ["Nová Ves"],
                            "latitude": 49.19,
                            "longitude": 16.6,
                            "district": "Okres Brno-venkov",
                            "region": "Jihomoravský kraj",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.directory = Directory(self.folder / "centers.json")

    def test_locality_without_a_center_returns_nearest_real_centers(self):
        result = self.directory.search("testovaci ves")
        self.assertEqual(result["ranking"], "straight_line_distance")
        self.assertEqual(result["centers"][0]["id"], 2)
        self.assertLess(result["centers"][0]["distance_km"], 2)
        distances = [c["distance_km"] for c in result["centers"]]
        self.assertEqual(distances, sorted(distances))
        self.assertIn("vzdušnou", result["note"])
        self.assertEqual(result["availability"], "unknown")

    def test_ambiguous_locality_requires_clarification(self):
        result = self.directory.search("Nova Ves")
        self.assertTrue(result["needs_clarification"])
        self.assertEqual(result["centers"], [])
        self.assertEqual(len(result["location_candidates"]), 2)
        resolved = self.directory.search("Nová Ves okres Brno-venkov")
        self.assertEqual(resolved["centers"][0]["id"], 1)

    def test_unknown_location_and_named_center_do_not_get_fake_distances(self):
        self.assertEqual(self.directory.search("Unknown Municipality")["centers"], [])
        named = self.directory.search("Testovací centrum Brno")
        self.assertEqual(named["centers"][0]["id"], 1)
        self.assertNotIn("distance_km", named["centers"][0])

    def test_missing_place_index_preserves_text_search(self):
        self.path.unlink()
        self.assertEqual(self.directory.search("trebic")["centers"][0]["id"], 2)
