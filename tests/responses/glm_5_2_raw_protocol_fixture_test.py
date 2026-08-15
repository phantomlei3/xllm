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

"""Characterization checks for the frozen GLM-5.2 raw protocol."""

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parents[1] / "fixtures/responses/model-protocol/glm-5.2.json"
REQUIRED_SCENARIOS = {
    "reasoning_text_stop",
    "reasoning_function_call",
    "parallel_function_calls",
    "reasoning_apply_patch",
    "function_output_continue",
    "custom_output_continue",
    "reasoning_truncated",
}


class Glm52RawProtocolFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_identity_and_coverage(self):
        self.assertEqual(self.fixture["schema_version"], 1)
        self.assertEqual(self.fixture["profile"], "glm_moe_dsa_responses")
        self.assertEqual(
            {case["scenario_id"] for case in self.fixture["scenarios"]},
            REQUIRED_SCENARIOS,
        )
        identity = self.fixture["identity"]
        for key in ("model", "tokenizer_sha256", "template_sha256", "runtime"):
            self.assertTrue(identity[key])

    def test_raw_tokens_text_segments_items_and_events_align(self):
        for case in self.fixture["scenarios"]:
            with self.subTest(case=case["scenario_id"]):
                raw = case["raw_generation"]
                self.assertEqual(
                    "".join(segment["raw"] for segment in case["segments"]),
                    raw["text"],
                )
                self.assertEqual(len(raw["token_ids"]), len(raw["token_text"]))
                self.assertEqual("".join(raw["token_text"]), raw["text"])
                self.assertTrue(case["expected_items"])
                self.assertEqual(case["events"][-1]["type"], case["terminal_event"])
                sequences = [event["sequence_number"] for event in case["events"]]
                self.assertEqual(sequences, list(range(len(sequences))))

    def test_native_control_tokens_and_patch_boundaries_are_frozen(self):
        grammar = self.fixture["grammar"]
        self.assertEqual(grammar["boundaries"]["tool_call_open"], "<tool_call>")
        self.assertEqual(grammar["boundaries"]["tool_call_close"], "</tool_call>")
        self.assertEqual(grammar["boundaries"]["argument_key_open"], "<arg_key>")
        self.assertEqual(grammar["boundaries"]["argument_value_open"], "<arg_value>")

        patch_case = next(case for case in self.fixture["scenarios"] if case["scenario_id"] == "reasoning_apply_patch")
        self.assertEqual(patch_case["repeat_count"], 2)
        self.assertEqual(
            patch_case["repeat_payload_sha256"][0],
            patch_case["repeat_payload_sha256"][1],
        )
        self.assertEqual(len(patch_case["repeat_raw_sha256"]), 2)
        self.assertIn(
            "<tool_call>apply_patch<arg_key>patch</arg_key><arg_value>",
            patch_case["raw_generation"]["text"],
        )
        self.assertEqual(
            patch_case["expected_items"][-1]["input"],
            "*** Begin Patch\n*** Add File: fixture.txt\n+hello\n*** End Patch\n",
        )

    def test_clear_thinking_false_preserves_all_reasoning(self):
        preserve = self.fixture["preserve_characterization"]
        self.assertEqual(preserve["required_control"], {"clear_thinking": False})
        tokens = preserve["preserved_prompt_token_ids"]
        for subsequence in preserve["reasoning_token_subsequences"]:
            self.assertTrue(
                any(tokens[i : i + len(subsequence)] == subsequence for i in range(len(tokens) - len(subsequence) + 1))
            )
        self.assertNotEqual(
            preserve["cleared_prompt_token_ids"],
            preserve["preserved_prompt_token_ids"],
        )

    def test_tool_outputs_continue_and_length_stop_is_incomplete(self):
        cases = {case["scenario_id"]: case for case in self.fixture["scenarios"]}
        function_types = {item.get("type") for item in cases["function_output_continue"]["canonical_request"]["input"]}
        custom_types = {item.get("type") for item in cases["custom_output_continue"]["canonical_request"]["input"]}
        self.assertIn("function_call_output", function_types)
        self.assertIn("custom_tool_call_output", custom_types)

        truncated = cases["reasoning_truncated"]
        self.assertEqual(truncated["raw_generation"]["finish_reason"], "length")
        self.assertEqual(truncated["terminal_event"], "response.incomplete")
        self.assertEqual(
            truncated["incomplete_details"],
            {"reason": "max_output_tokens"},
        )


if __name__ == "__main__":
    unittest.main()
