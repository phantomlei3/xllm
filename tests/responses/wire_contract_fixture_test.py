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

APPLY_PATCH_GRAMMAR = """start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""


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

    def test_apply_patch_tool_matches_captured_codex_definition(self):
        for fixture in (
            "requests/codex-tool-loop-start.json",
            "requests/apply-patch-stream.json",
        ):
            request = self.load_json(fixture)
            apply_patch = next(tool for tool in request["tools"] if tool["name"] == "apply_patch")
            self.assertEqual(
                apply_patch["description"],
                "Apply a patch in the fixture workspace.",
            )
            self.assertEqual(
                apply_patch["format"],
                {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": APPLY_PATCH_GRAMMAR,
                },
            )

    def test_codex_no_effect_fields_have_direct_request_evidence(self):
        request = self.load_json("requests/codex-tool-loop-start.json")
        self.assertEqual(request["reasoning"]["summary"], "auto")
        self.assertEqual(request["include"], ["reasoning.encrypted_content"])
        self.assertEqual(request["prompt_cache_key"], "fixture-session")
        self.assertEqual(request["text"], {"verbosity": "low"})
        self.assertEqual(
            request["client_metadata"],
            {"x-codex-turn-metadata": "<redacted>"},
        )

    def test_matrix_covers_known_codex_and_schema_branches(self):
        matrix = self.load_json("compatibility-matrix.json")
        expected = {
            "request_fields": {"presence_penalty", "frequency_penalty"},
            "input_items": {
                "reasoning.encrypted_content:null",
                "reasoning.summary:[]",
                "additional_tools",
                "image_generation_call",
            },
            "content_parts": {"refusal"},
            "tools": {
                "custom.description",
                "custom.format.grammar",
                "custom.other",
            },
        }
        for category, required_names in expected.items():
            names = {entry["name"] for entry in matrix["categories"][category]}
            self.assertLessEqual(required_names, names, category)
        for entries in matrix["categories"].values():
            for entry in entries:
                if entry["state"] == "accepted_no_effect":
                    self.assertTrue((FIXTURE_DIR / entry["evidence"]).is_file())

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
