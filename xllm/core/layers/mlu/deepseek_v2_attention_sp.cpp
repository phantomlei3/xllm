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

#include <tuple>

#include "deepseek_v2_attention.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

torch::Tensor DeepseekV2AttentionImpl::forward_sp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_sp::DeepseekV32SPContext& sp_ctx,
    KVCache& kv_cache,
    bool is_prefill_or_chunked_prefill) {
  CHECK(can_use_sp())
      << "deepseek_v32 sequence parallel requires replicated attention "
         "weights and lighting indexer.";
  CHECK(is_prefill_or_chunked_prefill)
      << "deepseek_v32 sequence parallel only supports prefill batches.";
  auto k_cache_scale = kv_cache.get_k_cache_scale();
  auto query_prep = prep_query(hidden_states, full_heads());
  if (attn_metadata.is_prefill && !use_prefill_mqa_) {
    return forward_sp_prefill_mha(
        positions, hidden_states, attn_metadata, sp_ctx, kv_cache, query_prep);
  }

  std::optional<torch::Tensor> new_block_tables = std::nullopt;
  std::optional<torch::Tensor> new_context_lens = std::nullopt;
  v32_sp::PaddedGatherHandle mla_handle;
  torch::Tensor index_cache = kv_cache.get_index_cache();
  IndexerSPPreOut index_pre;
  v32_sp::PaddedGatherHandle index_handle;

  Device device(hidden_states.device());
  if (sp_comm_stream_ == nullptr) {
    sp_comm_stream_ = device.get_stream_from_pool();
  }
  index_pre = indexer_->sp_pre(hidden_states,
                               query_prep.q_norm,
                               positions,
                               sp_ctx.local_attn_metadata,
                               sp_ctx);
  auto compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    index_handle = indexer_->sp_comm(index_pre.k_local, sp_ctx);
  }

  auto mla_inputs =
      build_sp_mla_inputs(hidden_states, positions, query_prep, sp_ctx);

  torch::Tensor k_gathered =
      indexer_->sp_wait_k(index_pre.k_local, index_handle, sp_ctx);
  compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    mla_handle = sp_mla_comm(mla_inputs.k_input, sp_ctx);
  }
  auto index_out = indexer_->sp_post(index_pre,
                                     k_gathered,
                                     index_cache,
                                     attn_metadata,
                                     sp_ctx.gathered_slot_mapping,
                                     sp_ctx);
  new_block_tables = std::get<0>(index_out);
  new_context_lens = std::get<1>(index_out);
  finish_sp_k_gather(mla_inputs, mla_handle, sp_ctx);

  AttentionMetadata attn_indexer_metadata =
      build_mla_attention_metadata(positions,
                                   hidden_states,
                                   mla_inputs.q_norm,
                                   mla_inputs.k_input,
                                   attn_metadata,
                                   kv_cache,
                                   k_cache_scale,
                                   is_prefill_or_chunked_prefill,
                                   sp_ctx.gathered_slot_mapping,
                                   new_block_tables,
                                   new_context_lens);
  attn_indexer_metadata.q_cu_seq_lens =
      sp_ctx.local_attn_metadata.q_cu_seq_lens;
  attn_indexer_metadata.max_query_len =
      sp_ctx.local_attn_metadata.max_query_len;
  auto [attn_output_local, output_lse] = attn_(attn_indexer_metadata,
                                               mla_inputs.q_input,
                                               mla_inputs.k_input,
                                               mla_inputs.v_input,
                                               kv_cache);
  return project_output(attn_output_local, full_heads());
}

torch::Tensor DeepseekV2AttentionImpl::forward_sp_prefill_mha(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    const v32_sp::DeepseekV32SPContext& sp_ctx,
    KVCache& kv_cache,
    const QueryPrep& query_prep) {
  auto k_cache_scale = kv_cache.get_k_cache_scale();

  std::optional<torch::Tensor> new_block_tables = std::nullopt;
  std::optional<torch::Tensor> new_context_lens = std::nullopt;
  v32_sp::PaddedGatherHandle mla_handle;
  torch::Tensor index_cache = kv_cache.get_index_cache();
  IndexerSPPreOut index_pre;
  v32_sp::PaddedGatherHandle index_handle;

  Device device(hidden_states.device());
  if (sp_comm_stream_ == nullptr) {
    sp_comm_stream_ = device.get_stream_from_pool();
  }
  index_pre = indexer_->sp_pre(hidden_states,
                               query_prep.q_norm,
                               positions,
                               sp_ctx.local_attn_metadata,
                               sp_ctx);
  auto compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    index_handle = indexer_->sp_comm(index_pre.k_local, sp_ctx);
  }

  PrefillMha prefill_mha =
      build_sp_prefill_mha(hidden_states, positions, query_prep, sp_ctx);

  torch::Tensor k_gathered =
      indexer_->sp_wait_k(index_pre.k_local, index_handle, sp_ctx);
  compute_stream = device.current_stream();
  sp_comm_stream_->wait_stream(*compute_stream);
  {
    torch::StreamGuard stream_guard = sp_comm_stream_->set_stream_guard();
    mla_handle = sp_mla_comm(prefill_mha.cache_input, sp_ctx);
  }
  auto index_out = indexer_->sp_post(index_pre,
                                     k_gathered,
                                     index_cache,
                                     attn_metadata,
                                     sp_ctx.gathered_slot_mapping,
                                     sp_ctx);
  new_block_tables = std::get<0>(index_out);
  new_context_lens = std::get<1>(index_out);
  torch::Tensor cache_gathered = parallel_state::finish_gather(mla_handle);
  torch::Tensor cache_global =
      v32_sp::restore_gathered_to_global_order(cache_gathered, sp_ctx);
  MhaKv kv = build_mha_kv(cache_global, full_heads());
  torch::Tensor k_full =
      kv.k_input.reshape({-1, full_heads().attn, qk_head_dim_});
  torch::Tensor v_full =
      kv.v_input.reshape({-1, full_heads().attn, v_head_dim_});

  std::vector<torch::Tensor> k_segments;
  std::vector<torch::Tensor> v_segments;
  std::vector<int32_t> ctx_lens;
  k_segments.reserve(sp_ctx.local_segments.size());
  v_segments.reserve(sp_ctx.local_segments.size());
  ctx_lens.reserve(sp_ctx.local_segments.size());
  int32_t max_ctx_len = 0;
  for (const auto& segment : sp_ctx.local_segments) {
    const int32_t ctx_start = sp_ctx.req_ctx_offsets_cpu.at(segment.req_idx);
    const int32_t ctx_end = ctx_start + segment.ctx_k_len;
    k_segments.push_back(k_full.slice(0, ctx_start, ctx_end));
    v_segments.push_back(v_full.slice(0, ctx_start, ctx_end));
    ctx_lens.push_back(segment.ctx_k_len);
    max_ctx_len = std::max(max_ctx_len, segment.ctx_k_len);
  }
  torch::Tensor k_packed = torch::cat(k_segments, 0);
  torch::Tensor v_packed = torch::cat(v_segments, 0);
  prefill_mha.k_input = k_packed.reshape({k_packed.size(0), -1});
  prefill_mha.v_input = v_packed.reshape({v_packed.size(0), -1});

  build_mla_attention_metadata(positions,
                               hidden_states,
                               prefill_mha.q_norm,
                               cache_gathered,
                               attn_metadata,
                               kv_cache,
                               k_cache_scale,
                               /*is_prefill_phase=*/true,
                               sp_ctx.gathered_slot_mapping,
                               new_block_tables,
                               new_context_lens);

  AttentionMetadata prefill_metadata = attn_metadata;
  prefill_metadata.q_cu_seq_lens = sp_ctx.local_attn_metadata.q_cu_seq_lens;
  prefill_metadata.kv_cu_seq_lens =
      v32_sp::make_sp_prefix(
          ctx_lens,
          torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU))
          .to(prefill_metadata.q_cu_seq_lens.device());
  prefill_metadata.max_query_len = sp_ctx.local_attn_metadata.max_query_len;
  prefill_metadata.max_seq_len = max_ctx_len;
  torch::Tensor attn_output =
      run_prefill_mha(prefill_mha, prefill_metadata, full_heads());
  return o_proj_->forward(attn_output);
}

DeepseekV2AttentionImpl::MlaInputs DeepseekV2AttentionImpl::build_sp_mla_inputs(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const QueryPrep& query_prep,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  MlaInputs out;
  out.q_input = torch::empty({hidden_states.size(0),
                              full_heads().attn,
                              kv_lora_rank_ + qk_rope_head_dim_},
                             hidden_states.options());
  out.q_norm = query_prep.q_norm;
  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);
  fill_q_input(out.q_input,
               query_prep.q,
               positions,
               sp_ctx.local_attn_metadata,
               /*use_prompt_rope=*/false);
  decode_kv_pre_base(latent_cache,
                     positions,
                     sp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);
  out.v_input = latent_cache.slice(-1, 0, kv_lora_rank_);
  out.k_input = latent_cache;
  out.q_input = out.q_input.view({out.q_input.size(0), -1});
  out.k_input = out.k_input.view({out.k_input.size(0), -1});
  out.v_input = out.v_input.view({out.v_input.size(0), -1});
  return out;
}

DeepseekV2AttentionImpl::PrefillMha
DeepseekV2AttentionImpl::build_sp_prefill_mha(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const QueryPrep& query_prep,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  const int32_t dim = -1;
  PrefillMha out;
  out.q_norm = query_prep.q_norm;

  auto q_vec = query_prep.q.split({qk_nope_head_dim_, qk_rope_head_dim_}, dim);
  torch::Tensor q_nope = q_vec[0];
  torch::Tensor q_pe = q_vec[1];
  rotary_emb_->forward(q_pe,
                       positions,
                       sp_ctx.local_attn_metadata.q_cu_seq_lens,
                       sp_ctx.local_attn_metadata.max_query_len,
                       /*is_prompt=*/false);
  torch::Tensor q = torch::empty_like(query_prep.q);
  q.slice(dim, 0, qk_nope_head_dim_).copy_(q_nope);
  q.slice(dim, qk_nope_head_dim_).copy_(q_pe);

  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);
  decode_kv_pre_base(latent_cache,
                     positions,
                     sp_ctx.local_attn_metadata,
                     /*use_prompt_rope=*/false);

  out.q_input = q.reshape({q.size(0), -1});
  out.cache_input = latent_cache.reshape({latent_cache.size(0), -1});
  return out;
}

v32_sp::PaddedGatherHandle DeepseekV2AttentionImpl::sp_mla_comm(
    const torch::Tensor& k_input,
    const v32_sp::DeepseekV32SPContext& sp_ctx) const {
  return parallel_state::launch_gather(
      k_input, sp_ctx.process_group, sp_ctx.comm_plan.tokens_per_rank);
}

void DeepseekV2AttentionImpl::finish_sp_k_gather(
    MlaInputs& mla_inputs,
    const v32_sp::PaddedGatherHandle& k_handle,
    const v32_sp::DeepseekV32SPContext& sp_ctx) const {
  (void)sp_ctx;
  mla_inputs.k_input = parallel_state::finish_gather(k_handle);
  mla_inputs.v_input = mla_inputs.k_input.slice(-1, 0, kv_lora_rank_);
}

}  // namespace layer
}  // namespace xllm
