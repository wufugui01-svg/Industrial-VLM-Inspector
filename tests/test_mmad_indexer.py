"""Tests for the MMAD JSONL index builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.mmad_indexer import build_mmad_index

REQUIRED_FIELDS = {
    "sample_id",
    "image_path",
    "image_relative_path",
    "image_exists",
    "question",
    "answer",
    "answer_text",
    "options",
    "task_type",
    "object_category",
}


class MMADIndexerTest(unittest.TestCase):
    def _make_fixture(self, root: Path) -> None:
        image_path = root / "ExampleSet" / "widget" / "test" / "bad" / "001.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(b"not-a-real-image")

        payload = {
            "ExampleSet/widget/test/bad/001.png": {
                "conversation": [
                    {
                        "Question": "Is there a defect?",
                        "Answer": "A",
                        "Options": {"A": "Yes", "B": "No"},
                        "type": "Anomaly Detection",
                    },
                    {},
                ]
            },
            "ExampleSet/gadget/test/bad/missing.png": {
                "conversation": [
                    {
                        "Question": "Where is the defect?",
                        "Answer": "B",
                        "Options": ["A: left", "B: right"],
                        "type": "Defect Localization",
                    }
                ]
            },
        }
        (root / "mmad.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_writes_jsonl_with_complete_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "MMAD"
            root.mkdir()
            self._make_fixture(root)
            output = Path(temporary_directory) / "output" / "index.jsonl"

            summary = build_mmad_index(
                mmad_root=root,
                output_path=output,
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertEqual(summary.total_samples, 3)
            self.assertEqual(summary.missing_image_count, 1)
            self.assertTrue(all(REQUIRED_FIELDS <= row.keys() for row in rows))
            self.assertTrue(all(Path(row["image_path"]).is_absolute() for row in rows))
            self.assertEqual(
                rows[0]["image_relative_path"],
                "ExampleSet/widget/test/bad/001.png",
            )
            self.assertEqual(rows[0]["answer_text"], "Yes")

    def test_missing_fields_use_defaults_and_limit_is_applied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "MMAD"
            root.mkdir()
            self._make_fixture(root)
            output = Path(temporary_directory) / "index.jsonl"

            summary = build_mmad_index(
                mmad_root=root,
                output_path=output,
                limit=2,
            )
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary.total_samples, 2)
            self.assertEqual(rows[1]["question"], "")
            self.assertEqual(rows[1]["answer"], "")
            self.assertEqual(rows[1]["options"], [])
            self.assertEqual(rows[1]["task_type"], "unknown")
            self.assertEqual(rows[1]["object_category"], "widget")


if __name__ == "__main__":
    unittest.main()
