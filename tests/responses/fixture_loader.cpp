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

#include "responses/fixture_loader.h"

#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string_view>
#include <utility>

namespace xllm::testing {
namespace {

constexpr int32_t kSchemaVersion = 1;
const std::set<std::string> kRequiredScenarios = {
    "reasoning_text_stop",
    "reasoning_function_call",
    "parallel_function_calls",
    "reasoning_apply_patch",
    "function_output_continue",
    "custom_output_continue",
    "reasoning_truncated",
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw FixtureError("fixture is not readable: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

nlohmann::json load_json(const std::filesystem::path& path) {
  const std::string contents = read_file(path);
  try {
    return nlohmann::json::parse(contents);
  } catch (const nlohmann::json::exception& error) {
    throw FixtureError("invalid JSON or UTF-8 in " + path.string() + ": " +
                       error.what());
  }
}

bool valid_utf8(const std::string& text) {
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t first = static_cast<uint8_t>(text[index]);
    if (first <= 0x7f) {
      ++index;
      continue;
    }
    size_t continuation_count = 0;
    uint32_t code_point = 0;
    uint32_t minimum = 0;
    if ((first & 0xe0) == 0xc0) {
      continuation_count = 1;
      code_point = first & 0x1f;
      minimum = 0x80;
    } else if ((first & 0xf0) == 0xe0) {
      continuation_count = 2;
      code_point = first & 0x0f;
      minimum = 0x800;
    } else if ((first & 0xf8) == 0xf0) {
      continuation_count = 3;
      code_point = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (index + continuation_count >= text.size()) {
      return false;
    }
    for (size_t offset = 1; offset <= continuation_count; ++offset) {
      const uint8_t byte = static_cast<uint8_t>(text[index + offset]);
      if ((byte & 0xc0) != 0x80) {
        return false;
      }
      code_point = (code_point << 6) | (byte & 0x3f);
    }
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff)) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

void validate_fixture_files(const std::filesystem::path& root) {
  constexpr std::string_view sensitive_markers[] = {
      "-----BEGIN PRIVATE KEY-----",
      "Authorization: Bearer ",
      "AKIA",
      "sk-",
  };
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string contents = read_file(entry.path());
    if (!valid_utf8(contents)) {
      throw FixtureError("invalid UTF-8 in fixture: " + entry.path().string());
    }
    for (std::string_view marker : sensitive_markers) {
      if (contents.find(marker) != std::string::npos) {
        throw FixtureError("sensitive data marker in fixture: " +
                           entry.path().string());
      }
    }
  }
}

void require_fields(const nlohmann::json& value,
                    std::initializer_list<const char*> fields,
                    const std::string& context) {
  if (!value.is_object()) {
    throw FixtureError("expected object in " + context);
  }
  for (const char* field : fields) {
    if (!value.contains(field)) {
      throw FixtureError("missing field '" + std::string(field) + "' in " +
                         context);
    }
  }
}

void validate_schema(const nlohmann::json& fixture,
                     const std::filesystem::path& path) {
  require_fields(fixture, {"schema_version"}, path.string());
  if (!fixture["schema_version"].is_number_integer() ||
      fixture["schema_version"].get<int32_t>() != kSchemaVersion) {
    throw FixtureError("unsupported fixture schema version: " + path.string());
  }
}

void validate_scenario(const nlohmann::json& scenario,
                       const std::string& context) {
  require_fields(scenario,
                 {"scenario_id",
                  "canonical_request",
                  "final_prompt",
                  "prompt_token_ids",
                  "raw_generation",
                  "segments",
                  "expected_items",
                  "events",
                  "terminal_event"},
                 context);
  const nlohmann::json& raw = scenario["raw_generation"];
  require_fields(raw, {"token_ids", "text", "finish_reason"}, context);
  if (!raw["token_ids"].is_array() || !raw["text"].is_string()) {
    throw FixtureError("invalid raw generation fields in " + context);
  }
  if (raw.contains("token_text")) {
    if (!raw["token_text"].is_array() ||
        raw["token_text"].size() != raw["token_ids"].size()) {
      throw FixtureError("token/text mismatch in " + context);
    }
    std::string decoded;
    for (const nlohmann::json& token_text : raw["token_text"]) {
      if (!token_text.is_string()) {
        throw FixtureError("token/text mismatch in " + context);
      }
      decoded += token_text.get_ref<const std::string&>();
    }
    if (decoded != raw["text"].get_ref<const std::string&>()) {
      throw FixtureError("token/text mismatch in " + context);
    }
  }
  if (!scenario["segments"].is_array()) {
    throw FixtureError("invalid segments in " + context);
  }
  std::string segmented;
  for (const nlohmann::json& segment : scenario["segments"]) {
    require_fields(segment, {"kind", "raw"}, context + " segment");
    if (!segment["raw"].is_string()) {
      throw FixtureError("invalid segment raw text in " + context);
    }
    segmented += segment["raw"].get_ref<const std::string&>();
  }
  if (segmented != raw["text"].get_ref<const std::string&>()) {
    throw FixtureError("segment/text mismatch in " + context);
  }
  if (!scenario["events"].is_array() || scenario["events"].empty()) {
    throw FixtureError("missing events in " + context);
  }
  size_t sequence_number = 0;
  for (const nlohmann::json& event : scenario["events"]) {
    require_fields(event, {"type", "sequence_number"}, context + " event");
    if (!event["sequence_number"].is_number_unsigned() ||
        event["sequence_number"].get<size_t>() != sequence_number) {
      throw FixtureError("event sequence mismatch in " + context);
    }
    ++sequence_number;
  }
  if (!scenario["terminal_event"].is_string() ||
      scenario["events"].back()["type"] != scenario["terminal_event"]) {
    throw FixtureError("terminal event mismatch in " + context);
  }
}

std::string scenario_key(const std::string& profile,
                         const std::string& scenario_id) {
  return profile + "\n" + scenario_id;
}

const char* artifact_key(ExpectedArtifact artifact) {
  switch (artifact) {
    case ExpectedArtifact::PREPARED_REQUEST:
      return "expected_prepared_request";
    case ExpectedArtifact::PROMPT:
      return "final_prompt";
    case ExpectedArtifact::PROMPT_TOKEN_IDS:
      return "prompt_token_ids";
    case ExpectedArtifact::SEGMENTS:
      return "segments";
    case ExpectedArtifact::ITEMS:
      return "expected_items";
    case ExpectedArtifact::FINAL_RESPONSE:
      return "expected_final_response";
    case ExpectedArtifact::SSE_EVENTS:
      return "events";
  }
  throw FixtureError("unknown expected artifact");
}

std::string wire_key(const std::string& scenario_id,
                     ExpectedArtifact artifact) {
  return scenario_id + "\n" + artifact_key(artifact);
}

std::vector<nlohmann::json> load_sse(const std::filesystem::path& path) {
  std::istringstream input(read_file(path));
  std::vector<nlohmann::json> events;
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("data: ", 0) != 0) {
      continue;
    }
    try {
      events.emplace_back(nlohmann::json::parse(line.substr(6)));
    } catch (const nlohmann::json::exception& error) {
      throw FixtureError("invalid SSE data in " + path.string() + ": " +
                         error.what());
    }
  }
  if (events.empty()) {
    throw FixtureError("SSE fixture has no data events: " + path.string());
  }
  return events;
}

}  // namespace

FixtureError::FixtureError(const std::string& message)
    : std::runtime_error(message) {}

FixtureCatalog FixtureCatalog::load(const std::filesystem::path& root) {
  validate_fixture_files(root);
  FixtureCatalog catalog;
  const std::filesystem::path model_root = root / "model-protocol";
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(model_root)) {
    if (entry.path().extension() != ".json") {
      continue;
    }
    nlohmann::json fixture = load_json(entry.path());
    validate_schema(fixture, entry.path());
    require_fields(fixture,
                   {"profile",
                    "identity",
                    "grammar",
                    "preserve_characterization",
                    "scenarios"},
                   entry.path().string());
    const std::string profile = fixture.at("profile").get<std::string>();
    if (!catalog.profiles_.emplace(profile).second) {
      throw FixtureError("duplicate profile: " + profile);
    }
    std::set<std::string>& ids = catalog.scenario_ids_[profile];
    for (const nlohmann::json& scenario : fixture.at("scenarios")) {
      validate_scenario(scenario, entry.path().string());
      const std::string scenario_id =
          scenario.at("scenario_id").get<std::string>();
      if (!ids.emplace(scenario_id).second) {
        throw FixtureError("duplicate scenario ID: " + scenario_id);
      }
      catalog.scenarios_.emplace(scenario_key(profile, scenario_id), scenario);
    }
    if (ids != kRequiredScenarios) {
      throw FixtureError(
          "model fixture does not contain all required scenarios: " + profile);
    }
  }

  const std::filesystem::path wire_root = root / "wire" / "codex-cli-0.147.0";
  const nlohmann::json manifest = load_json(wire_root / "manifest.json");
  validate_schema(manifest, wire_root / "manifest.json");
  require_fields(manifest,
                 {"fixture_family",
                  "codex_cli_version",
                  "freeze_date",
                  "traffic_coverage",
                  "fixtures"},
                 (wire_root / "manifest.json").string());
  for (const nlohmann::json& fixture : manifest.at("fixtures")) {
    require_fields(fixture,
                   {"path", "kind", "source"},
                   (wire_root / "manifest.json").string());
    const std::string path = fixture.at("path").get<std::string>();
    const std::filesystem::path full_path = wire_root / path;
    if (full_path.extension() == ".sse") {
      catalog.sse_.emplace(path, load_sse(full_path));
    } else {
      catalog.wire_.emplace(path, load_json(full_path));
    }
  }
  const std::filesystem::path wire_scenarios_path =
      wire_root / "wire-scenarios.json";
  const nlohmann::json wire_scenarios = load_json(wire_scenarios_path);
  validate_schema(wire_scenarios, wire_scenarios_path);
  require_fields(wire_scenarios,
                 {"fixture_family", "scenarios"},
                 wire_scenarios_path.string());
  for (const nlohmann::json& scenario : wire_scenarios["scenarios"]) {
    require_fields(scenario,
                   {"scenario_id", "input", "expected"},
                   wire_scenarios_path.string());
    const std::string scenario_id = scenario["scenario_id"].get<std::string>();
    const std::string input_path = scenario["input"].get<std::string>();
    if (!catalog.wire_inputs_.emplace(scenario_id, catalog.wire_.at(input_path))
             .second) {
      throw FixtureError("duplicate wire scenario ID: " + scenario_id);
    }
    for (const auto& [artifact_name, expected_path] :
         scenario["expected"].items()) {
      ExpectedArtifact artifact;
      if (artifact_name == "prepared_request") {
        artifact = ExpectedArtifact::PREPARED_REQUEST;
      } else if (artifact_name == "final_response") {
        artifact = ExpectedArtifact::FINAL_RESPONSE;
      } else if (artifact_name == "sse_events") {
        artifact = ExpectedArtifact::SSE_EVENTS;
      } else {
        throw FixtureError("unknown wire expected artifact: " + artifact_name);
      }
      const std::string path = expected_path.get<std::string>();
      nlohmann::json value;
      if (artifact == ExpectedArtifact::SSE_EVENTS) {
        value = catalog.sse_.at(path);
      } else {
        value = catalog.wire_.at(path);
      }
      catalog.wire_expected_.emplace(wire_key(scenario_id, artifact),
                                     std::move(value));
    }
  }
  catalog.baseline_ = load_json(root / "baseline/chat-route.json");
  validate_schema(catalog.baseline_, root / "baseline/chat-route.json");
  require_fields(catalog.baseline_,
                 {"fixture_family", "route", "reasoning_parser", "templates"},
                 (root / "baseline/chat-route.json").string());
  return catalog;
}

const std::set<std::string>& FixtureCatalog::profiles() const {
  return profiles_;
}

const std::set<std::string>& FixtureCatalog::scenario_ids(
    const std::string& profile) const {
  return scenario_ids_.at(profile);
}

const nlohmann::json& FixtureCatalog::input(
    const std::string& profile,
    const std::string& scenario_id) const {
  return scenarios_.at(scenario_key(profile, scenario_id))
      .at("canonical_request");
}

const nlohmann::json& FixtureCatalog::expected(
    const std::string& profile,
    const std::string& scenario_id,
    ExpectedArtifact artifact) const {
  return scenarios_.at(scenario_key(profile, scenario_id))
      .at(artifact_key(artifact));
}

const nlohmann::json& FixtureCatalog::wire(
    const std::string& relative_path) const {
  return wire_.at(relative_path);
}

const nlohmann::json& FixtureCatalog::wire_input(
    const std::string& scenario_id) const {
  return wire_inputs_.at(scenario_id);
}

const nlohmann::json& FixtureCatalog::wire_expected(
    const std::string& scenario_id,
    ExpectedArtifact artifact) const {
  return wire_expected_.at(wire_key(scenario_id, artifact));
}

const std::vector<nlohmann::json>& FixtureCatalog::sse(
    const std::string& relative_path) const {
  return sse_.at(relative_path);
}

const nlohmann::json& FixtureCatalog::baseline() const { return baseline_; }

}  // namespace xllm::testing
