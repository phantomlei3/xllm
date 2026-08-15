/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <string>

#include "responses/fixture_loader.h"

namespace xllm::testing {
namespace {

const std::filesystem::path kFixtureRoot =
    std::filesystem::path(XLLM_SOURCE_DIR) / "tests/fixtures/responses";

class FixtureTree final {
 public:
  FixtureTree() {
    root_ = std::filesystem::temp_directory_path() /
            "xllm-responses-fixture-conformance";
    std::filesystem::remove_all(root_);
    std::filesystem::copy(
        kFixtureRoot, root_, std::filesystem::copy_options::recursive);
  }

  ~FixtureTree() { std::filesystem::remove_all(root_); }

  const std::filesystem::path& root() const { return root_; }

  void edit_json(const std::string& relative_path,
                 const std::function<void(nlohmann::json&)>& edit) const {
    const std::filesystem::path path = root_ / relative_path;
    std::ifstream input(path);
    nlohmann::json fixture = nlohmann::json::parse(input);
    input.close();
    edit(fixture);
    std::ofstream output(path, std::ios::trunc);
    output << fixture.dump(/*indent=*/2) << '\n';
  }

 private:
  std::filesystem::path root_;
};

void expect_load_error(const std::filesystem::path& root,
                       const std::string& message_part) {
  try {
    FixtureCatalog::load(root);
    FAIL() << "expected FixtureError containing: " << message_part;
  } catch (const FixtureError& error) {
    EXPECT_NE(std::string(error.what()).find(message_part), std::string::npos)
        << error.what();
  }
}

TEST(ResponsesFixtureCatalog, DiscoversEveryRequiredModelScenarioOnce) {
  FixtureCatalog catalog = FixtureCatalog::load(kFixtureRoot);
  const std::set<std::string> required = {
      "reasoning_text_stop",
      "reasoning_function_call",
      "parallel_function_calls",
      "reasoning_apply_patch",
      "function_output_continue",
      "custom_output_continue",
      "reasoning_truncated",
  };

  EXPECT_EQ(catalog.profiles(),
            std::set<std::string>(
                {"deepseek_v4_responses", "glm_moe_dsa_responses"}));
  for (const std::string& profile : catalog.profiles()) {
    EXPECT_EQ(catalog.scenario_ids(profile), required);
  }
}

TEST(ResponsesFixtureCatalog, SelectsExpectedArtifactsForOneInput) {
  FixtureCatalog catalog = FixtureCatalog::load(kFixtureRoot);
  const std::string profile = "deepseek_v4_responses";
  const std::string scenario_id = "reasoning_apply_patch";

  EXPECT_TRUE(catalog.input(profile, scenario_id).contains("tools"));
  EXPECT_TRUE(catalog.expected(profile, scenario_id, ExpectedArtifact::PROMPT)
                  .is_string());
  EXPECT_TRUE(
      catalog.expected(profile, scenario_id, ExpectedArtifact::PROMPT_TOKEN_IDS)
          .is_array());
  EXPECT_TRUE(catalog.expected(profile, scenario_id, ExpectedArtifact::SEGMENTS)
                  .is_array());
  EXPECT_TRUE(catalog.expected(profile, scenario_id, ExpectedArtifact::ITEMS)
                  .is_array());
  EXPECT_TRUE(
      catalog.expected(profile, scenario_id, ExpectedArtifact::SSE_EVENTS)
          .is_array());
}

TEST(ResponsesFixtureCatalog, LoadsWireAndBaselineGolden) {
  FixtureCatalog catalog = FixtureCatalog::load(kFixtureRoot);

  EXPECT_EQ(catalog.wire("requests/plain-text-nonstream.json")["stream"],
            false);
  EXPECT_EQ(catalog.wire("expected-responses/plain-text.json")["object"],
            "response");
  EXPECT_EQ(catalog.sse("expected-sse/tool-loop.sse").back()["type"],
            "response.completed");
  EXPECT_EQ(catalog.baseline()["route"]["path"], "/v1/chat/completions");
}

TEST(ResponsesFixtureCatalog, BaselineMatchesCurrentChatRouteAndTemplates) {
  FixtureCatalog catalog = FixtureCatalog::load(kFixtureRoot);
  const nlohmann::json& baseline = catalog.baseline();
  const auto read_source = [](const std::string& relative_path) {
    std::ifstream input(std::filesystem::path(XLLM_SOURCE_DIR) / relative_path);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  };

  EXPECT_NE(read_source("xllm/server/xllm_server.cpp")
                .find("v1/chat/completions => ChatCompletionsHttp"),
            std::string::npos);
  EXPECT_NE(read_source("xllm/proto/xllm_service.proto")
                .find("rpc " + baseline["route"]["rpc"].get<std::string>()),
            std::string::npos);
  EXPECT_NE(read_source("tests/core/framework/chat_template/"
                        "deepseek_v4_cpp_template_test.cpp")
                .find(baseline["templates"]["deepseek_v4_tool_marker"]
                          .get<std::string>()),
            std::string::npos);
  EXPECT_NE(
      read_source("xllm/core/framework/chat_template/jinja_chat_template.cpp")
          .find(baseline["templates"]["jinja_reasoning_field"]
                    .get<std::string>()),
      std::string::npos);
}

TEST(ResponsesFixtureCatalog, SelectsWireExpectationsByScenario) {
  FixtureCatalog catalog = FixtureCatalog::load(kFixtureRoot);

  EXPECT_EQ(catalog.wire_input("plain_text_nonstream")["input"],
            "Return one short fixture sentence.");
  EXPECT_EQ(catalog.wire_expected(
                "plain_text_nonstream",
                ExpectedArtifact::PREPARED_REQUEST)["sequence_count"],
            1);
  EXPECT_EQ(catalog.wire_expected("plain_text_nonstream",
                                  ExpectedArtifact::FINAL_RESPONSE)["status"],
            "completed");
  EXPECT_EQ(catalog.wire_expected("tool_loop", ExpectedArtifact::SSE_EVENTS)
                .back()["type"],
            "response.completed");
}

TEST(ResponsesFixtureCatalog, RejectsMissingFieldAndUnknownVersion) {
  FixtureTree fixture_tree;
  fixture_tree.edit_json(
      "model-protocol/deepseek-v4.json",
      [](nlohmann::json& fixture) { fixture.erase("identity"); });
  expect_load_error(fixture_tree.root(), "missing field");

  FixtureTree unknown_version;
  unknown_version.edit_json(
      "model-protocol/deepseek-v4.json",
      [](nlohmann::json& fixture) { fixture["schema_version"] = 2; });
  expect_load_error(unknown_version.root(), "schema version");
}

TEST(ResponsesFixtureCatalog, RejectsInvalidUtf8AndSensitiveData) {
  FixtureTree invalid_utf8;
  const std::filesystem::path invalid_path =
      invalid_utf8.root() / "model-protocol/deepseek-v4.json";
  std::string contents;
  {
    std::ifstream input(invalid_path, std::ios::binary);
    contents.assign(std::istreambuf_iterator<char>(input),
                    std::istreambuf_iterator<char>());
  }
  contents.insert(contents.find("deepseek_v4_responses"), "\xff");
  {
    std::ofstream output(invalid_path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
  }
  expect_load_error(invalid_utf8.root(), "UTF-8");

  FixtureTree sensitive;
  sensitive.edit_json("model-protocol/deepseek-v4.json",
                      [](nlohmann::json& fixture) {
                        fixture["identity"]["runtime"]["credential"] =
                            "sk-1234567890abcdefghijklmnop";
                      });
  expect_load_error(sensitive.root(), "sensitive data");
}

TEST(ResponsesFixtureCatalog, RejectsTokenTextAndSegmentMisalignment) {
  FixtureTree token_mismatch;
  token_mismatch.edit_json(
      "model-protocol/glm-5.2.json", [](nlohmann::json& fixture) {
        fixture["scenarios"][0]["raw_generation"]["token_text"][0] = "wrong";
      });
  expect_load_error(token_mismatch.root(), "token/text mismatch");

  FixtureTree segment_mismatch;
  segment_mismatch.edit_json(
      "model-protocol/deepseek-v4.json", [](nlohmann::json& fixture) {
        fixture["scenarios"][0]["segments"][0]["raw"] = "wrong";
      });
  expect_load_error(segment_mismatch.root(), "segment/text mismatch");
}

TEST(ResponsesFixtureCatalog, RejectsEventSequenceMisalignment) {
  FixtureTree fixture_tree;
  fixture_tree.edit_json(
      "model-protocol/deepseek-v4.json", [](nlohmann::json& fixture) {
        fixture["scenarios"][0]["events"][1]["sequence_number"] = 9;
      });
  expect_load_error(fixture_tree.root(), "event sequence mismatch");
}

TEST(ResponsesFixtureCatalog, RejectsDuplicateAndMissingRequiredScenarios) {
  FixtureTree duplicate;
  duplicate.edit_json("model-protocol/deepseek-v4.json",
                      [](nlohmann::json& fixture) {
                        fixture["scenarios"].push_back(fixture["scenarios"][0]);
                      });
  expect_load_error(duplicate.root(), "duplicate scenario ID");

  FixtureTree missing;
  missing.edit_json("model-protocol/glm-5.2.json", [](nlohmann::json& fixture) {
    fixture["scenarios"].erase(fixture["scenarios"].begin());
  });
  expect_load_error(missing.root(), "required scenarios");
}

TEST(ResponsesFixtureCatalog, NeverWritesStaticGolden) {
  const std::filesystem::path golden =
      kFixtureRoot / "model-protocol/deepseek-v4.json";
  const std::string before = [&golden]() {
    std::ifstream input(golden, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  }();
  const std::filesystem::file_time_type write_time =
      std::filesystem::last_write_time(golden);

  FixtureCatalog::load(kFixtureRoot);

  std::ifstream input(golden, std::ios::binary);
  const std::string after{std::istreambuf_iterator<char>(input),
                          std::istreambuf_iterator<char>()};
  EXPECT_EQ(after, before);
  EXPECT_EQ(std::filesystem::last_write_time(golden), write_time);
}

}  // namespace
}  // namespace xllm::testing
