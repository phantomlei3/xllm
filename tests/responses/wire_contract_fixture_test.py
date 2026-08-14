# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from pathlib import Path

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "responses" / "wire" / "codex-cli-0.147.0"


class WireContractFixtureTest(unittest.TestCase):
    def load_json(self, relative_path):
        with (FIXTURE_DIR / relative_path).open(encoding="utf-8") as fixture_file:
            return json.load(fixture_file)

    def test_matrix_entries_have_unique_state_and_evidence(self):
        matrix = self.load_json("compatibility-matrix.json")
        allowed_states = {
            "supported",
            "accepted_fixed",
            "accepted_no_effect",
            "unsupported",
        }
        for category, entries in matrix["categories"].items():
            names = [entry["name"] for entry in entries]
            self.assertEqual(len(names), len(set(names)), category)
            for entry in entries:
                self.assertIn(entry["state"], allowed_states)
                self.assertTrue(entry["semantics"])
                self.assertTrue(entry["error_behavior"])
                self.assertTrue(entry["evidence"])

    def test_manifest_covers_required_codex_traffic(self):
        manifest = self.load_json("manifest.json")
        self.assertEqual(manifest["codex_cli_version"], "0.147.0")
        self.assertEqual(manifest["freeze_date"], "2026-08-14")
        self.assertEqual(
            set(manifest["traffic_coverage"]),
            {
                "plain_text",
                "function_tool",
                "apply_patch",
                "function_tool_output",
                "custom_tool_output",
                "reasoning_replay",
                "non_stream",
                "stream",
            },
        )
        for fixture in manifest["fixtures"]:
            self.assertTrue((FIXTURE_DIR / fixture["path"]).is_file())
            self.assertTrue(fixture["source"])

    def test_captured_tool_loop_freezes_required_contract(self):
        request = self.load_json("requests/tool-loop-replay.json")
        self.assertFalse(request["store"])
        self.assertTrue(request["stream"])
        self.assertTrue(request["parallel_tool_calls"])
        self.assertEqual(request["tool_choice"], "auto")
        self.assertIn({"type": "custom", "name": "apply_patch"}, request["tools"])
        item_types = [item["type"] for item in request["input"]]
        for item_type in (
            "reasoning",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
        ):
            self.assertIn(item_type, item_types)

    def test_error_fixtures_freeze_fail_closed_constraints(self):
        errors = self.load_json("errors/rejected-requests.json")
        self.assertEqual(
            {case["expected_error"]["code"] for case in errors["cases"]},
            {
                "unsupported_parameter",
                "unsupported_content_type",
                "unknown_tool",
            },
        )


if __name__ == "__main__":
    unittest.main()
