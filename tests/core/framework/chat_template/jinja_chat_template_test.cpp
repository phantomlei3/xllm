/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>

namespace xllm {

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
};

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

TestableJinjaChatTemplate make_glm_encoder() {
  const std::filesystem::path root = XLLM_SOURCE_DIR;
  TokenizerArgs args;
  args.chat_template(read_file(
      root / "tests/fixtures/responses/model-protocol/glm-5.2-template.jinja"));
  args.bos_token("");
  args.eos_token("");
  return TestableJinjaChatTemplate(args);
}

std::string fixture_prompt(const std::string& scenario_id) {
  const std::filesystem::path path =
      std::filesystem::path(XLLM_SOURCE_DIR) /
      "tests/fixtures/responses/model-protocol/glm-5.2.json";
  std::ifstream input(path);
  const nlohmann::json fixture = nlohmann::json::parse(input);
  for (const nlohmann::json& scenario : fixture["scenarios"]) {
    if (scenario["scenario_id"] == scenario_id) {
      return scenario["final_prompt"].get<std::string>();
    }
  }
  return "";
}

FunctionTool weather_tool() {
  return {.name = "get_weather",
          .description = "Get the weather for a city.",
          .parameters = {{"type", "object"},
                         {"properties", {{"city", {{"type", "string"}}}}},
                         {"required", {"city"}}}};
}

CustomTool patch_tool() { return {.name = "apply_patch"}; }

std::optional<std::string> render_glm(
    const TestableJinjaChatTemplate& encoder,
    const ChatMessages& messages,
    const std::vector<Tool>& tools = {},
    model_protocol::ReasoningEffort effort =
        model_protocol::ReasoningEffort::HIGH,
    const model_protocol::ToolChoice& choice = {}) {
  return encoder.apply(
      messages,
      /*json_tools=*/{},
      tools,
      {.thinking_history = model_protocol::ThinkingHistoryPolicy::PRESERVE,
       .clear_thinking = false},
      effort,
      choice,
      /*chat_template_kwargs=*/{{"clear_thinking", true}});
}

}  // namespace

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, AppliesChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}"
      "{{ message['role'] + ': ' + message['content'] }}"
      "{% endfor %}"
      "{% if not enable_thinking %}<no_think>{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "user"}, {"content", "describe this image"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate template_(args);
  const nlohmann::ordered_json tools = nlohmann::json::array();
  auto result = template_.apply(messages, tools, chat_template_kwargs);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), "user: describe this image<no_think>");
}

TEST(JinjaChatTemplate, PreservePolicyOverridesClientThinkingControls) {
  const std::string template_str =
      "{{ clear_thinking }}|{{ enable_thinking }}|{{ reasoning_effort }}|"
      "{% for message in messages %}"
      "{{ message.role }}:{{ message.reasoning_content | default('') }};"
      "{% endfor %}";
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "First question.");
  Message assistant("assistant", "First answer.");
  assistant.reasoning_content = "historical reasoning";
  messages.emplace_back(std::move(assistant));
  messages.emplace_back("user", "Final question.");

  auto prompt = encoder.apply(
      messages,
      /*json_tools=*/{},
      /*protocol_tools=*/{},
      {.thinking_history = model_protocol::ThinkingHistoryPolicy::PRESERVE,
       .clear_thinking = false},
      model_protocol::ReasoningEffort::NONE,
      model_protocol::ToolChoice{},
      /*chat_template_kwargs=*/
      {{"clear_thinking", true},
       {"enable_thinking", true},
       {"reasoning_effort", "max"}});
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt,
            "False|False|none|user:;assistant:historical reasoning;user:;");
}

TEST(JinjaChatTemplate, TemplateDefaultKeepsClientThinkingControls) {
  const std::string template_str =
      "{{ clear_thinking }}|{{ enable_thinking }}|{{ reasoning_effort }}";
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate encoder(args);

  auto prompt = encoder.apply(
      /*messages=*/{},
      /*json_tools=*/{},
      /*protocol_tools=*/{},
      {.thinking_history =
           model_protocol::ThinkingHistoryPolicy::TEMPLATE_DEFAULT},
      model_protocol::ReasoningEffort::NONE,
      model_protocol::ToolChoice{},
      /*chat_template_kwargs=*/
      {{"clear_thinking", true},
       {"enable_thinking", true},
       {"reasoning_effort", "client"}});
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt, "True|True|client");
}

TEST(JinjaChatTemplate, RendersTypedToolsCallsAndOutputsForGlm) {
  const std::string template_str =
      "{% for tool in tools %}{{ tool.function | tojson }};{% endfor %}|"
      "{% for message in messages %}"
      "{% if message.tool_calls %}{{ message.tool_calls | tojson }}{% endif %}"
      "{% if message.role == 'tool' %}OUT={{ message.content }}{% endif %}"
      "{% endfor %}";
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate encoder(args);

  ChatMessages messages;
  messages.emplace_back("user", "Run both.");
  Message assistant("assistant", "");
  assistant.protocol_tool_calls = Message::ProtocolToolCallVec{
      FunctionCall{.id = "call_a",
                   .name = "get_weather",
                   .arguments = R"({"city":"Beijing"})"},
      CustomToolCall{.id = "call_b",
                     .name = "apply_patch",
                     .input = "*** Begin Patch\n+hello\n*** End Patch\n"}};
  messages.emplace_back(std::move(assistant));
  Message output("tool", "Done!");
  output.tool_call_id = "call_b";
  output.tool_output_kind = ToolOutputKind::CUSTOM;
  messages.emplace_back(std::move(output));

  std::vector<Tool> tools = {
      FunctionTool{.name = "get_weather",
                   .description = "Get the weather for a city.",
                   .parameters = {{"type", "object"}}},
      CustomTool{.name = "apply_patch"}};
  auto prompt = encoder.apply(
      messages,
      /*json_tools=*/{},
      tools,
      {.thinking_history = model_protocol::ThinkingHistoryPolicy::PRESERVE,
       .clear_thinking = false},
      model_protocol::ReasoningEffort::HIGH,
      {.kind = model_protocol::ToolChoiceKind::REQUIRED},
      /*chat_template_kwargs=*/{});
  ASSERT_TRUE(prompt.has_value());

  EXPECT_NE(prompt->find("\"name\": \"get_weather\""), std::string::npos);
  EXPECT_NE(prompt->find("Apply an exact patch"), std::string::npos);
  EXPECT_NE(prompt->find("\"arguments\": {\"city\": \"Beijing\"}"),
            std::string::npos);
  EXPECT_NE(prompt->find("\"arguments\": {\"patch\":"), std::string::npos);
  EXPECT_NE(prompt->find("OUT=Done!"), std::string::npos);
}

TEST(JinjaChatTemplate, NamedChoiceOnlyExposesSelectedTypedTool) {
  const std::string template_str =
      "{% for tool in tools %}{{ tool.function.name }};{% endfor %}";
  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  TestableJinjaChatTemplate encoder(args);

  std::vector<Tool> tools = {FunctionTool{.name = "first"},
                             FunctionTool{.name = "second"}};
  auto prompt = encoder.apply(
      /*messages=*/{},
      /*json_tools=*/{},
      tools,
      {.thinking_history = model_protocol::ThinkingHistoryPolicy::PRESERVE,
       .clear_thinking = false},
      model_protocol::ReasoningEffort::HIGH,
      {.kind = model_protocol::ToolChoiceKind::FUNCTION, .name = "second"},
      /*chat_template_kwargs=*/{});
  ASSERT_TRUE(prompt.has_value());

  EXPECT_EQ(*prompt, "second;");
}

TEST(JinjaChatTemplate, GlmMatchesFrozenTextPrompts) {
  TestableJinjaChatTemplate encoder = make_glm_encoder();

  auto text = render_glm(
      encoder, {Message("user", "Think briefly, then answer exactly: FOUR")});
  ASSERT_TRUE(text.has_value());
  EXPECT_EQ(*text, fixture_prompt("reasoning_text_stop"));

  auto truncated = render_glm(
      encoder,
      {Message("user", "Explain why the sky appears blue in detail.")});
  ASSERT_TRUE(truncated.has_value());
  EXPECT_EQ(*truncated, fixture_prompt("reasoning_truncated"));
}

TEST(JinjaChatTemplate, GlmMatchesFrozenFunctionPrompts) {
  TestableJinjaChatTemplate encoder = make_glm_encoder();
  const std::vector<Tool> tools = {weather_tool()};
  ChatMessages messages = {Message(
      "user",
      "Use get_weather exactly once for Beijing. Do not answer directly.")};

  auto initial = render_glm(encoder,
                            messages,
                            tools,
                            model_protocol::ReasoningEffort::HIGH,
                            {.kind = model_protocol::ToolChoiceKind::REQUIRED});
  ASSERT_TRUE(initial.has_value());
  EXPECT_EQ(*initial, fixture_prompt("reasoning_function_call"));

  Message assistant("assistant", "");
  assistant.reasoning_content =
      "The user wants me to call get_weather for Beijing exactly once, and "
      "not answer directly.";
  assistant.protocol_tool_calls = Message::ProtocolToolCallVec{
      FunctionCall{.id = "call_history_1",
                   .name = "get_weather",
                   .arguments = R"({"city":"Beijing"})"}};
  messages.emplace_back(std::move(assistant));
  Message output("tool", R"({"condition":"sunny","celsius":22})");
  output.tool_call_id = "call_history_1";
  output.tool_output_kind = ToolOutputKind::FUNCTION;
  messages.emplace_back(std::move(output));

  auto continued = render_glm(encoder, messages, tools);
  ASSERT_TRUE(continued.has_value());
  EXPECT_EQ(*continued, fixture_prompt("function_output_continue"));
}

TEST(JinjaChatTemplate, GlmMatchesFrozenParallelPrompt) {
  TestableJinjaChatTemplate encoder = make_glm_encoder();
  const std::vector<Tool> tools = {
      weather_tool(),
      FunctionTool{
          .name = "get_time",
          .description = "Get the time for a zone.",
          .parameters = {{"type", "object"},
                         {"properties", {{"zone", {{"type", "string"}}}}},
                         {"required", {"zone"}}}}};
  ChatMessages messages = {
      Message("user",
              "Call both get_weather for Beijing and get_time for UTC in one "
              "response. Do not answer directly.")};

  auto prompt = render_glm(encoder,
                           messages,
                           tools,
                           model_protocol::ReasoningEffort::HIGH,
                           {.kind = model_protocol::ToolChoiceKind::REQUIRED});
  ASSERT_TRUE(prompt.has_value());
  EXPECT_EQ(*prompt, fixture_prompt("parallel_function_calls"));
}

TEST(JinjaChatTemplate, GlmMatchesFrozenApplyPatchPrompts) {
  TestableJinjaChatTemplate encoder = make_glm_encoder();
  const std::vector<Tool> tools = {patch_tool()};
  const std::string user =
      "Call apply_patch exactly once. Set patch to this exact literal text, "
      "preserving the leading + and final newline:\n"
      "*** Begin Patch\n*** Add File: fixture.txt\n+hello\n*** End Patch\n"
      "Do not answer directly.";
  ChatMessages messages = {Message("user", user)};

  auto initial = render_glm(
      encoder,
      messages,
      tools,
      model_protocol::ReasoningEffort::HIGH,
      {.kind = model_protocol::ToolChoiceKind::CUSTOM, .name = "apply_patch"});
  ASSERT_TRUE(initial.has_value());
  EXPECT_EQ(*initial, fixture_prompt("reasoning_apply_patch"));

  Message assistant("assistant", "");
  assistant.reasoning_content =
      "The user wants me to call apply_patch with a specific patch text. Let "
      "me construct the exact patch text they provided, preserving the "
      "leading + and final newline.";
  assistant.protocol_tool_calls = Message::ProtocolToolCallVec{
      CustomToolCall{.id = "call_history_patch",
                     .name = "apply_patch",
                     .input = "*** Begin Patch\n*** Add File: "
                              "fixture.txt\n+hello\n*** End Patch\n"}};
  messages.emplace_back(std::move(assistant));
  Message output("tool", "Done!");
  output.tool_call_id = "call_history_patch";
  output.tool_output_kind = ToolOutputKind::CUSTOM;
  messages.emplace_back(std::move(output));

  auto continued = render_glm(encoder, messages, tools);
  ASSERT_TRUE(continued.has_value());
  EXPECT_EQ(*continued, fixture_prompt("custom_output_continue"));
}

TEST(JinjaChatTemplate, GlmPreservesAllReasoningWhenEffortIsNone) {
  TestableJinjaChatTemplate encoder = make_glm_encoder();
  ChatMessages messages;
  messages.emplace_back("user", "First question.");
  Message first("assistant", "First answer.");
  first.reasoning_content = "First private reasoning.";
  messages.emplace_back(std::move(first));
  messages.emplace_back("user", "Second question.");
  Message second("assistant", "Second answer.");
  second.reasoning_content = "Second private reasoning.";
  messages.emplace_back(std::move(second));
  messages.emplace_back("user", "Final question.");

  auto prompt = render_glm(encoder,
                           messages,
                           /*tools=*/{},
                           model_protocol::ReasoningEffort::NONE);
  ASSERT_TRUE(prompt.has_value());
  EXPECT_NE(prompt->find("<think>First private reasoning.</think>"),
            std::string::npos);
  EXPECT_NE(prompt->find("<think>Second private reasoning.</think>"),
            std::string::npos);
  EXPECT_EQ(prompt->find("Reasoning Effort:"), std::string::npos);
  EXPECT_TRUE(prompt->ends_with("<|assistant|><think></think>"));
}

}  // namespace xllm
