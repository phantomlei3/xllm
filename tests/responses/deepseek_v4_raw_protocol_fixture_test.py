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

"""Characterization checks for the frozen DeepSeek V4 raw protocol."""

import hashlib
import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parents[1] / "fixtures/responses/model-protocol/deepseek-v4.json"
REQUIRED_SCENARIOS = {
    "reasoning_text_stop",
    "reasoning_function_call",
    "parallel_function_calls",
    "reasoning_apply_patch",
    "function_output_continue",
    "custom_output_continue",
    "reasoning_truncated",
}


class DeepseekV4RawProtocolFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_identity_and_coverage(self):
        self.assertEqual(self.fixture["schema_version"], 1)
        self.assertEqual(self.fixture["profile"], "deepseek_v4_responses")
        self.assertEqual(
            {case["scenario_id"] for case in self.fixture["scenarios"]},
            REQUIRED_SCENARIOS,
        )
        identity = self.fixture["identity"]
        for key in ("model", "tokenizer_sha256", "template_sha256", "runtime"):
            self.assertTrue(identity[key])

    def test_template_fingerprint_matches_source(self) -> None:
        identity = self.fixture["identity"]
        template = Path(__file__).parents[2] / identity["template"]
        self.assertEqual(
            hashlib.sha256(template.read_bytes()).hexdigest(),
            identity["template_sha256"],
        )

    def test_raw_tokens_text_segments_items_and_events_align(self):
        for case in self.fixture["scenarios"]:
            with self.subTest(case=case["scenario_id"]):
                raw = case["raw_generation"]
                self.assertEqual(
                    "".join(segment["raw"] for segment in case["segments"]),
                    raw["text"],
                )
                self.assertTrue(case["expected_items"])
                self.assertEqual(case["events"][-1]["type"], case["terminal_event"])
                sequences = [event["sequence_number"] for event in case["events"]]
                self.assertEqual(sequences, list(range(len(sequences))))

    def test_native_control_tokens_and_payload_boundaries_are_frozen(self):
        grammar = self.fixture["grammar"]
        self.assertEqual(grammar["special_tokens"]["thinking_start"], [128821])
        self.assertEqual(grammar["special_tokens"]["thinking_end"], [128822])
        self.assertEqual(grammar["special_tokens"]["dsml"], [128825])
        self.assertEqual(grammar["special_tokens"]["eos"], [1])
        self.assertEqual(grammar["max_control_marker"]["token_length"], 6)

        patch_case = next(case for case in self.fixture["scenarios"] if case["scenario_id"] == "reasoning_apply_patch")
        self.assertEqual(patch_case["repeat_count"], 2)
        self.assertEqual(
            patch_case["repeat_raw_sha256"][0],
            patch_case["repeat_raw_sha256"][1],
        )
        self.assertIn(
            '<｜DSML｜invoke name="apply_patch">',
            patch_case["raw_generation"]["text"],
        )
        self.assertEqual(
            patch_case["expected_items"][-1]["input"],
            "*** Begin Patch\n*** Add File: fixture.txt\n+hello\n*** End Patch\n",
        )

    def test_preserve_disables_drop_thinking(self):
        preserve = self.fixture["preserve_characterization"]
        self.assertEqual(preserve["required_control"], {"drop_thinking": False})
        for subsequence in preserve["reasoning_token_subsequences"]:
            tokens = preserve["preserved_prompt_token_ids"]
            self.assertTrue(
                any(tokens[i : i + len(subsequence)] == subsequence for i in range(len(tokens) - len(subsequence) + 1))
            )
        self.assertNotEqual(
            preserve["dropped_prompt_token_ids"],
            preserve["preserved_prompt_token_ids"],
        )


if __name__ == "__main__":
    unittest.main()
