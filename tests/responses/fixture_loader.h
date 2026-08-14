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

#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm::testing {

enum class ExpectedArtifact {
  RAW_GENERATION,
  PREPARED_REQUEST,
  PROMPT,
  PROMPT_TOKEN_IDS,
  SEGMENTS,
  ITEMS,
  FINAL_RESPONSE,
  SSE_EVENTS,
};

class FixtureError final : public std::runtime_error {
 public:
  explicit FixtureError(const std::string& message);
};

class FixtureCatalog final {
 public:
  static FixtureCatalog load(const std::filesystem::path& root);

  const std::set<std::string>& profiles() const;
  const std::set<std::string>& scenario_ids(const std::string& profile) const;
  const nlohmann::json& input(const std::string& profile,
                              const std::string& scenario_id) const;
  const nlohmann::json& expected(const std::string& profile,
                                 const std::string& scenario_id,
                                 ExpectedArtifact artifact) const;
  const nlohmann::json& wire(const std::string& relative_path) const;
  const nlohmann::json& wire_input(const std::string& scenario_id) const;
  const nlohmann::json& wire_expected(const std::string& scenario_id,
                                      ExpectedArtifact artifact) const;
  const std::vector<nlohmann::json>& sse(
      const std::string& relative_path) const;
  const nlohmann::json& baseline() const;

 private:
  std::set<std::string> profiles_;
  std::unordered_map<std::string, std::set<std::string>> scenario_ids_;
  std::unordered_map<std::string, nlohmann::json> scenarios_;
  std::unordered_map<std::string, nlohmann::json> wire_;
  std::unordered_map<std::string, nlohmann::json> wire_inputs_;
  std::unordered_map<std::string, nlohmann::json> wire_expected_;
  std::unordered_map<std::string, std::vector<nlohmann::json>> sse_;
  nlohmann::json baseline_;
};

}  // namespace xllm::testing
