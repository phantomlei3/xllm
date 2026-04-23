/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "runtime/mtp_worker_impl.h"

#include <folly/futures/Future.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "common/global_flags.h"
#include "platform/device.h"

namespace xllm {
namespace {

std::vector<int32_t> tensor_to_vec_int32(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor =
      tensor.to(torch::kCPU).to(torch::kInt).contiguous();
  const int32_t* data = cpu_tensor.data_ptr<int32_t>();
  return {data, data + cpu_tensor.numel()};
}

std::vector<float> tensor_to_vec_float(const torch::Tensor& tensor) {
  torch::Tensor cpu_tensor =
      tensor.to(torch::kCPU).to(torch::kFloat32).contiguous();
  const float* data = cpu_tensor.data_ptr<float>();
  return {data, data + cpu_tensor.numel()};
}

ForwardInput make_decode_input() {
  ForwardInput input;
  input.token_ids = torch::tensor({101, 202, 303}, torch::kInt);
  input.positions = torch::tensor({1, 1, 1}, torch::kInt);
  input.input_params.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.num_sequences = 3;
  input.input_params.embedding_ids = {0, 1, 2};
  input.input_params.kv_max_seq_len = 2;
  input.input_params.q_max_seq_len = 1;
  input.input_params.kv_seq_lens_vec = {0, 2, 4, 6};
  input.input_params.q_seq_lens_vec = {0, 1, 2, 3};
  input.input_params.kv_seq_lens =
      torch::tensor(input.input_params.kv_seq_lens_vec, torch::kInt);
  input.input_params.q_seq_lens =
      torch::tensor(input.input_params.q_seq_lens_vec, torch::kInt);
  input.input_params.q_cu_seq_lens = torch::tensor({1, 2, 3}, torch::kInt);
  input.input_params.block_tables = torch::tensor({{0}, {1}, {2}}, torch::kInt);
  input.input_params.new_cache_slots =
      torch::tensor({0, 128, 256}, torch::kInt);
  input.sampling_params.selected_token_idxes =
      torch::tensor({0, 1, 2}, torch::kInt);
  input.sampling_params.sample_idxes = torch::tensor({0, 1, 2}, torch::kInt);
  return input;
}

ForwardInput make_decode_input_two_rows() {
  ForwardInput input;
  input.token_ids = torch::tensor({101, 202}, torch::kInt);
  input.positions = torch::tensor({1, 1}, torch::kInt);
  input.input_params.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.num_sequences = 2;
  input.input_params.embedding_ids = {0, 1};
  input.input_params.kv_max_seq_len = 2;
  input.input_params.q_max_seq_len = 1;
  input.input_params.kv_seq_lens_vec = {0, 2, 4};
  input.input_params.q_seq_lens_vec = {0, 1, 2};
  input.input_params.kv_seq_lens =
      torch::tensor(input.input_params.kv_seq_lens_vec, torch::kInt);
  input.input_params.q_seq_lens =
      torch::tensor(input.input_params.q_seq_lens_vec, torch::kInt);
  input.input_params.q_cu_seq_lens = torch::tensor({1, 2}, torch::kInt);
  input.input_params.block_tables = torch::tensor({{0}, {1}}, torch::kInt);
  input.input_params.new_cache_slots = torch::tensor({0, 128}, torch::kInt);
  input.sampling_params.selected_token_idxes =
      torch::tensor({0, 1}, torch::kInt);
  input.sampling_params.sample_idxes = torch::tensor({0, 1}, torch::kInt);
  return input;
}

ForwardOutput make_target_output_two_rows() {
  ForwardOutput output;
  output.logits = torch::tensor({{0.1f, 0.2f, 0.3f, 1.4f, -0.4f, 0.0f},
                                 {-0.3f, 0.5f, 0.1f, 0.2f, 1.3f, -0.8f},
                                 {0.0f, 1.1f, 0.3f, -0.2f, -0.4f, 0.2f},
                                 {0.4f, 0.3f, 1.0f, -0.7f, -0.1f, 0.0f}},
                                torch::kFloat32);
  output.sample_output.next_tokens = torch::tensor({3, 4, 1, 2}, torch::kInt);
  output.sample_output.embeddings =
      torch::tensor({{10.0f, 11.0f, 12.0f, 13.0f},
                     {20.0f, 21.0f, 22.0f, 23.0f},
                     {30.0f, 31.0f, 32.0f, 33.0f},
                     {40.0f, 41.0f, 42.0f, 43.0f}},
                    torch::kFloat32);
  output.sample_output.top_tokens =
      torch::tensor({{3, 2, 1}, {4, 3, 2}, {1, 0, 2}, {2, 1, 3}}, torch::kLong);
  output.sample_output.top_logprobs = torch::tensor({{-0.1f, -0.2f, -0.3f},
                                                     {-0.4f, -0.5f, -0.6f},
                                                     {-0.7f, -0.8f, -0.9f},
                                                     {-1.0f, -1.1f, -1.2f}},
                                                    torch::kFloat32);
  return output;
}

KVCacheShape make_kv_cache_shape() {
  KVCacheCapacity kv_cache_cap;
  kv_cache_cap.n_blocks(4).block_size(16);

  ModelArgs model_args;
  model_args.model_type("joyai_llm_flash").n_heads(8).head_dim(16);

  return KVCacheShape(kv_cache_cap, model_args, /*world_size=*/1);
}

class FakeLLMWorkerImpl final : public LLMWorkerImpl {
 public:
  FakeLLMWorkerImpl(const torch::Device& device, const ForwardOutput& output)
      : LLMWorkerImpl(make_parallel_args(), device, make_options()),
        output_(output) {}

  std::optional<ForwardOutput> step(const ForwardInput& input) override {
    last_input_ = input;
    return output_;
  }

  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& input) override {
    last_input_ = input;
    folly::Promise<std::optional<ForwardOutput>> promise;
    auto future = promise.getSemiFuture();
    promise.setValue(output_);
    return future;
  }

  const std::optional<ForwardInput>& last_input() const { return last_input_; }

 private:
  static ParallelArgs make_parallel_args() {
    return ParallelArgs(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  }

  static runtime::Options make_options() {
    runtime::Options options;
    options.enable_schedule_overlap(false).num_speculative_tokens(1);
    return options;
  }

  ForwardOutput output_;
  std::optional<ForwardInput> last_input_;
};

#if !defined(USE_NPU)
class FakeTransferLLMWorkerImpl final : public LLMWorkerImpl {
 public:
  explicit FakeTransferLLMWorkerImpl(const torch::Device& device)
      : LLMWorkerImpl(make_parallel_args(), device, make_options()) {
    status_ = WorkerImpl::Status::LOADED;
  }

  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override {
    transfer_calls_ += 1;
    last_num_blocks_ = kv_cache_shape.key_cache_shape()[0];
    status_ = WorkerImpl::Status::READY;
    return true;
  }

#if defined(USE_MLU)
  bool allocate_kv_cache_with_transfer(
      std::shared_ptr<KVCacheTransfer> kv_cache_transfer,
      const KVCacheShape& kv_cache_shape) override {
    (void)kv_cache_transfer;
    return allocate_kv_cache_with_transfer(kv_cache_shape);
  }
#endif

  std::optional<ForwardOutput> step(const ForwardInput& input) override {
    (void)input;
    return std::nullopt;
  }

  int32_t transfer_calls() const { return transfer_calls_; }

  int64_t last_num_blocks() const { return last_num_blocks_; }

 private:
  static ParallelArgs make_parallel_args() {
    return ParallelArgs(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  }

  static runtime::Options make_options() {
    runtime::Options options;
    options.enable_schedule_overlap(false).num_speculative_tokens(1);
    return options;
  }

  int32_t transfer_calls_ = 0;
  int64_t last_num_blocks_ = 0;
};
#endif

class TestMTPWorkerImpl final : public MTPWorkerImpl {
 public:
  explicit TestMTPWorkerImpl(const torch::Device& device)
      : MTPWorkerImpl(make_parallel_args(), device, make_options()) {
    dtype_ = torch::kFloat32;
    embedding_size_ = 4;
    embedding_cache_ = std::make_shared<EmbeddingCache>(8);
  }

  void write_cache(const std::vector<int32_t>& embedding_ids,
                   const torch::Tensor& next_tokens,
                   const torch::Tensor& embeddings,
                   const torch::Tensor& probs) {
    embedding_cache_->write(embedding_ids, next_tokens, embeddings, probs);
  }

  ForwardOutput prepare(const ForwardInput& input, torch::Tensor& miss_mask) {
    return prepare_last_output_for_decode(input, miss_mask);
  }

  std::optional<ForwardOutput> run_decode_single(const ForwardInput& input) {
    return step_decode_single(input);
  }

  std::optional<ForwardOutput> run_validate_for_test(
      const ForwardInput& input,
      const torch::Tensor& miss_mask,
      const std::vector<ForwardOutput>& draft_outputs,
      ForwardInput& validate_input) {
    return run_validate(input, miss_mask, draft_outputs, validate_input);
  }

  void set_target_impl(std::unique_ptr<LLMWorkerImpl> impl) {
    impl_ = std::move(impl);
  }

  void set_draft_impl(std::unique_ptr<LLMWorkerImpl> impl) {
    draft_impl_ = std::move(impl);
  }

  void set_validate_output(const SampleOutput& output) {
    validate_output_ = output;
  }

  void patch_rows(const ForwardOutput& target_output,
                  const torch::Tensor& miss_mask,
                  SampleOutput& sample_output) {
    patch_force_reject_rows(target_output, miss_mask, sample_output);
  }

  std::vector<uint8_t> build_seed_mask(const std::vector<int32_t>& ids) const {
    return embedding_cache_->build_seed_mask(ids);
  }

  ForwardOutput read_cache(const std::vector<int32_t>& ids) {
    return embedding_cache_->read_for_decode(ids);
  }

  std::vector<int32_t> read_correction_tokens(
      const std::vector<int32_t>& ids) const {
    return embedding_cache_->read_correction_tokens(ids);
  }

  std::vector<int32_t> read_position_offsets(
      const std::vector<int32_t>& ids) const {
    return embedding_cache_->read_position_offsets(ids);
  }

 protected:
  SampleOutput validate(const SamplingParameters& sampling_params,
                        const std::vector<ForwardOutput>& draft_outputs,
                        const ForwardOutput& target_output) override {
    (void)sampling_params;
    (void)draft_outputs;
    (void)target_output;
    return validate_output_;
  }

 private:
  static ParallelArgs make_parallel_args() {
    return ParallelArgs(
        /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  }

  static runtime::Options make_options() {
    runtime::Options options;
    options.enable_schedule_overlap(false).num_speculative_tokens(1);
    return options;
  }

  SampleOutput validate_output_;
};

TEST(MTPWorkerImplTest, PrepareLastOutputForDecodeBuildsDummyRows) {
  torch::Device device(Device::type_torch(), 0);
  TestMTPWorkerImpl worker(device);
  worker.write_cache(
      /*embedding_ids=*/{0, 2},
      torch::tensor({11, 33}, torch::kInt),
      torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}}),
      torch::tensor({0.25f, 0.75f}));

  ForwardInput input;
  input.token_ids = torch::tensor({101, 202, 303}, torch::kInt);
  input.input_params.num_sequences = 3;
  input.input_params.embedding_ids = {0, 1, 2};

  torch::Tensor miss_mask;
  ForwardOutput output = worker.prepare(input, miss_mask);

  EXPECT_EQ(output.sample_output.next_tokens.device(), device);
  EXPECT_EQ(output.sample_output.embeddings.device(), device);
  EXPECT_EQ(output.sample_output.probs.device(), device);
  EXPECT_EQ(miss_mask.device(), device);

  EXPECT_EQ(tensor_to_vec_int32(output.sample_output.next_tokens),
            (std::vector<int32_t>{11, 202, 33}));
  EXPECT_EQ(tensor_to_vec_int32(miss_mask.to(torch::kInt)),
            (std::vector<int32_t>{0, 1, 0}));

  torch::Tensor expected_embeddings = torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f},
                                                     {0.0f, 0.0f, 0.0f, 0.0f},
                                                     {5.0f, 6.0f, 7.0f, 8.0f}});
  EXPECT_TRUE(torch::equal(output.sample_output.embeddings.to(torch::kCPU),
                           expected_embeddings));
  EXPECT_EQ(tensor_to_vec_float(output.sample_output.probs),
            (std::vector<float>{0.25f, 1.0f, 0.75f}));
}

TEST(MTPWorkerImplTest, StepDecodeSingleWiresMixedPrepareIntoValidatePath) {
  torch::Device device(Device::type_torch(), 0);
  TestMTPWorkerImpl worker(device);
  worker.write_cache(
      /*embedding_ids=*/{0, 2},
      torch::tensor({11, 33}, torch::kInt),
      torch::tensor({{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}}),
      torch::tensor({0.25f, 0.75f}));

  SampleOutput validate_output;
  validate_output.next_tokens =
      torch::tensor({{501}, {502}, {503}}, torch::kInt);
  validate_output.embeddings = torch::tensor({{{1.0f, 1.0f, 1.0f, 1.0f}},
                                              {{2.0f, 2.0f, 2.0f, 2.0f}},
                                              {{3.0f, 3.0f, 3.0f, 3.0f}}});
  worker.set_validate_output(validate_output);

  ForwardOutput target_output;
  target_output.sample_output.next_tokens =
      torch::tensor({41, 42, 43}, torch::kInt);
  target_output.sample_output.embeddings =
      torch::tensor({{4.1f, 4.2f, 4.3f, 4.4f},
                     {5.1f, 5.2f, 5.3f, 5.4f},
                     {6.1f, 6.2f, 6.3f, 6.4f}},
                    torch::kFloat32);
  auto target_impl = std::make_unique<FakeLLMWorkerImpl>(device, target_output);
  FakeLLMWorkerImpl* target_impl_ptr = target_impl.get();

  ForwardOutput draft_output;
  draft_output.sample_output.next_tokens =
      torch::tensor({71, 72, 73}, torch::kInt);
  draft_output.sample_output.embeddings =
      torch::tensor({{9.0f, 9.0f, 9.0f, 9.0f},
                     {8.0f, 8.0f, 8.0f, 8.0f},
                     {7.0f, 7.0f, 7.0f, 7.0f}});
  draft_output.sample_output.probs = torch::tensor({0.6f, 0.7f, 0.8f});
  auto draft_impl = std::make_unique<FakeLLMWorkerImpl>(device, draft_output);

  worker.set_target_impl(std::move(target_impl));
  worker.set_draft_impl(std::move(draft_impl));

  std::optional<ForwardOutput> output_opt =
      worker.run_decode_single(make_decode_input());

  ASSERT_TRUE(output_opt.has_value());
  ASSERT_TRUE(target_impl_ptr->last_input().has_value());
  EXPECT_EQ(tensor_to_vec_int32(
                target_impl_ptr->last_input()->token_ids.to(torch::kCPU)),
            (std::vector<int32_t>{101, 11, 202, 202, 303, 33}));
  EXPECT_EQ(tensor_to_vec_int32(
                output_opt->sample_output.next_tokens.to(torch::kCPU)),
            (std::vector<int32_t>{501, 42, 503}));
}

#if !defined(USE_NPU)
TEST(MTPWorkerImplTest, AllocateKvCacheWithTransferUsesInnerWorkers) {
  torch::Device device(Device::type_torch(), 0);
  TestMTPWorkerImpl worker(device);
  auto target_impl = std::make_unique<FakeTransferLLMWorkerImpl>(device);
  auto draft_impl = std::make_unique<FakeTransferLLMWorkerImpl>(device);
  FakeTransferLLMWorkerImpl* target_impl_ptr = target_impl.get();
  FakeTransferLLMWorkerImpl* draft_impl_ptr = draft_impl.get();
#if defined(USE_MLU)
  const std::string saved_transfer_type = FLAGS_kv_cache_transfer_type;
  FLAGS_kv_cache_transfer_type = "Mooncake";
#endif

  worker.set_target_impl(std::move(target_impl));
  worker.set_draft_impl(std::move(draft_impl));

  bool allocated =
      worker.allocate_kv_cache_with_transfer(make_kv_cache_shape());

  EXPECT_TRUE(allocated);
  EXPECT_EQ(target_impl_ptr->transfer_calls(), 1);
  EXPECT_EQ(draft_impl_ptr->transfer_calls(), 1);
  EXPECT_EQ(target_impl_ptr->last_num_blocks(), 4);
  EXPECT_EQ(draft_impl_ptr->last_num_blocks(), 4);
  EXPECT_EQ(target_impl_ptr->get_status(), WorkerImpl::Status::READY);
  EXPECT_EQ(draft_impl_ptr->get_status(), WorkerImpl::Status::READY);
#if defined(USE_MLU)
  FLAGS_kv_cache_transfer_type = saved_transfer_type;
#endif
}
#endif

TEST(MTPWorkerImplTest, ValidateForceRejectRowsOnBootstrapMiss) {
  torch::Device device(Device::type_torch(), 0);
  TestMTPWorkerImpl worker(device);
  ForwardOutput target_output = make_target_output_two_rows();
  SampleOutput sample_output;
  sample_output.next_tokens = torch::tensor({{99, 98}, {77, 76}}, torch::kInt);
  sample_output.embeddings =
      torch::tensor({{{1.0f, 1.1f, 1.2f, 1.3f}, {2.0f, 2.1f, 2.2f, 2.3f}},
                     {{3.0f, 3.1f, 3.2f, 3.3f}, {4.0f, 4.1f, 4.2f, 4.3f}}},
                    torch::kFloat32);
  sample_output.logprobs =
      torch::tensor({{-9.0f, -8.0f}, {-7.0f, -6.0f}}, torch::kFloat32);
  sample_output.top_tokens = torch::tensor(
      {{{99, 98, 97}, {96, 95, 94}}, {{77, 76, 75}, {74, 73, 72}}},
      torch::kLong);
  sample_output.top_logprobs =
      torch::tensor({{{-9.0f, -8.0f, -7.0f}, {-6.0f, -5.0f, -4.0f}},
                     {{-3.0f, -2.0f, -1.0f}, {-0.5f, -0.4f, -0.3f}}},
                    torch::kFloat32);
  torch::Tensor miss_mask =
      torch::tensor({true, false}, torch::dtype(torch::kBool)).to(device);

  worker.patch_rows(target_output, miss_mask, sample_output);

  EXPECT_EQ(tensor_to_vec_int32(sample_output.next_tokens.to(torch::kCPU)),
            (std::vector<int32_t>{3, -1, 77, 76}));

  torch::Tensor expected_embeddings = torch::tensor(
      {{{10.0f, 11.0f, 12.0f, 13.0f}, {20.0f, 21.0f, 22.0f, 23.0f}},
       {{3.0f, 3.1f, 3.2f, 3.3f}, {4.0f, 4.1f, 4.2f, 4.3f}}},
      torch::kFloat32);
  EXPECT_TRUE(torch::equal(sample_output.embeddings.to(torch::kCPU),
                           expected_embeddings));

  torch::Tensor expected_logprobs =
      torch::log_softmax(target_output.logits.view({2, 2, 6}),
                         /*dim=*/-1,
                         /*dtype=*/torch::kFloat32);
  EXPECT_FLOAT_EQ(sample_output.logprobs[0][0].item<float>(),
                  expected_logprobs[0][0][3].item<float>());
  EXPECT_FLOAT_EQ(sample_output.logprobs[0][1].item<float>(), 0.0f);
  EXPECT_FLOAT_EQ(sample_output.logprobs[1][0].item<float>(), -7.0f);
  EXPECT_FLOAT_EQ(sample_output.logprobs[1][1].item<float>(), -6.0f);

  EXPECT_EQ(
      tensor_to_vec_int32(sample_output.top_tokens.to(torch::kCPU)),
      (std::vector<int32_t>{3, 2, 1, -1, -1, -1, 77, 76, 75, 74, 73, 72}));
  EXPECT_EQ(tensor_to_vec_float(sample_output.top_logprobs.to(torch::kCPU)),
            (std::vector<float>{-0.1f,
                                -0.2f,
                                -0.3f,
                                0.0f,
                                0.0f,
                                0.0f,
                                -3.0f,
                                -2.0f,
                                -1.0f,
                                -0.5f,
                                -0.4f,
                                -0.3f}));
}

TEST(MTPWorkerImplTest, RunDraftExtendRefillsSeedAfterForcedReject) {
  torch::Device device(Device::type_torch(), 0);
  TestMTPWorkerImpl worker(device);

  SampleOutput validate_output;
  validate_output.next_tokens =
      torch::tensor({{91, 92}, {777, 778}}, torch::kInt);
  validate_output.embeddings = torch::tensor(
      {{{90.0f, 90.1f, 90.2f, 90.3f}, {91.0f, 91.1f, 91.2f, 91.3f}},
       {{70.0f, 70.1f, 70.2f, 70.3f}, {71.0f, 71.1f, 71.2f, 71.3f}}},
      torch::kFloat32);
  worker.set_validate_output(validate_output);

  ForwardOutput target_output = make_target_output_two_rows();
  auto target_impl = std::make_unique<FakeLLMWorkerImpl>(device, target_output);

  ForwardOutput extend_output;
  extend_output.sample_output.next_tokens =
      torch::tensor({551, 552}, torch::kInt);
  extend_output.sample_output.embeddings = torch::tensor(
      {{1.5f, 1.6f, 1.7f, 1.8f}, {2.5f, 2.6f, 2.7f, 2.8f}}, torch::kFloat32);
  extend_output.sample_output.probs = torch::tensor({0.25f, 0.75f});
  auto draft_impl = std::make_unique<FakeLLMWorkerImpl>(device, extend_output);
  FakeLLMWorkerImpl* draft_impl_ptr = draft_impl.get();

  worker.set_target_impl(std::move(target_impl));
  worker.set_draft_impl(std::move(draft_impl));

  ForwardInput input = make_decode_input_two_rows();
  ForwardInput validate_input = input;
  std::vector<ForwardOutput> draft_outputs(1);
  draft_outputs[0].sample_output.next_tokens =
      torch::tensor({11, 12}, torch::kInt);
  torch::Tensor miss_mask =
      torch::tensor({true, false}, torch::dtype(torch::kBool)).to(device);

  std::optional<ForwardOutput> output_opt = worker.run_validate_for_test(
      input, miss_mask, draft_outputs, validate_input);

  ASSERT_TRUE(output_opt.has_value());
  ASSERT_TRUE(draft_impl_ptr->last_input().has_value());
  EXPECT_EQ(tensor_to_vec_int32(
                draft_impl_ptr->last_input()->token_ids.to(torch::kCPU)),
            (std::vector<int32_t>{101, 3, 777, 778}));
  EXPECT_EQ(tensor_to_vec_int32(output_opt->sample_output.next_tokens),
            (std::vector<int32_t>{3, -1, 777, 778}));
  EXPECT_EQ(worker.build_seed_mask({0, 1}), (std::vector<uint8_t>{1, 1}));
  EXPECT_EQ(worker.read_correction_tokens({0}), (std::vector<int32_t>{3}));
  EXPECT_EQ(worker.read_position_offsets({0}), (std::vector<int32_t>{0}));

  ForwardOutput cached_output = worker.read_cache({0});
  EXPECT_EQ(tensor_to_vec_int32(cached_output.sample_output.next_tokens),
            (std::vector<int32_t>{551}));
  EXPECT_TRUE(
      torch::equal(cached_output.sample_output.embeddings.to(torch::kCPU),
                   torch::tensor({{1.5f, 1.6f, 1.7f, 1.8f}}, torch::kFloat32)));
  EXPECT_EQ(
      tensor_to_vec_float(cached_output.sample_output.probs.to(torch::kCPU)),
      (std::vector<float>{0.25f}));
}

}  // namespace
}  // namespace xllm
